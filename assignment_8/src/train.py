import copy
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

criterion = torch.nn.CrossEntropyLoss()

def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train(gnn, loader, epochs=50):
    optimizer = optim.Adam(gnn.parameters(), lr=0.01)

    # Train for N epochs
    for epoch in range(epochs):
        loss = train_epoch(loader, gnn, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


def generate_pseudo_labels(model, unlabeled_loader, confidence_threshold=0.9):
    model.eval()
    pseudo_graphs = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(out, dim=1)
            confidence, predicted = probs.max(dim=1)

            for i in range(batch.num_graphs):
                if confidence[i] >= confidence_threshold:
                    mask = batch.batch == i
                    node_indices = mask.nonzero(as_tuple=True)[0]
                    edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                    local_edge_index = batch.edge_index[:, edge_mask]

                    idx_map = {old.item(): new for new, old in enumerate(node_indices)}
                    local_edge_index = torch.tensor(
                        [[idx_map[n.item()] for n in local_edge_index[0]],
                         [idx_map[n.item()] for n in local_edge_index[1]]],
                        dtype=torch.long
                    )
                    pseudo_graphs.append(Data(
                        x=batch.x[mask],
                        edge_index=local_edge_index,
                        y=predicted[i]
                    ))
    return pseudo_graphs


def train_with_pseudo_labels(gnn, labeled_graphs, loader, unlabeled_loader, epochs=50):
    optimizer = optim.Adam(gnn.parameters(), lr=0.01)
    combined_loader = loader

    for epoch in range(epochs):
        if epoch % 5 == 0 and epoch > 0:
            pseudo_graphs = generate_pseudo_labels(gnn, unlabeled_loader)
            combined_loader = DataLoader(labeled_graphs + pseudo_graphs, batch_size=32, shuffle=True)

        loss = train_epoch(combined_loader, gnn, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


def update_ema(student, teacher, alpha=0.99):
    """Copy student weights into teacher via exponential moving average."""
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data


def train_with_mean_teacher(gnn, labeled_loader, unlabeled_loader, epochs=50, alpha=0.99, consistency_weight=1.0):
    """
    Mean-Teacher semi-supervised training.

    - student (gnn): updated by gradient descent on labeled CE loss + consistency loss
    - teacher: EMA copy of student, never trained directly
    - consistency loss: MSE between student and teacher softmax outputs on unlabeled data
    """
    teacher = copy.deepcopy(gnn)
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(gnn.parameters(), lr=0.01)
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(epochs):
        gnn.train()
        teacher.train()  # keep BN/dropout in same mode as student
        total_loss = 0

        for labeled_batch in labeled_loader:
            optimizer.zero_grad()

            # --- supervised loss on labeled batch ---
            student_out = gnn(labeled_batch.x, labeled_batch.edge_index, labeled_batch.batch)
            sup_loss = criterion(student_out, labeled_batch.y)

            # --- consistency loss on unlabeled batch ---
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            student_unlabeled = gnn(unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.batch)
            with torch.no_grad():
                teacher_unlabeled = teacher(unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.batch)

            # MSE between softmax probabilities (standard Mean-Teacher consistency)
            consistency_loss = F.mse_loss(
                torch.softmax(student_unlabeled, dim=1),
                torch.softmax(teacher_unlabeled, dim=1)
            )

            loss = sup_loss + consistency_weight * consistency_loss
            loss.backward()
            optimizer.step()

            # update teacher weights after each student step
            update_ema(gnn, teacher, alpha=alpha)

            total_loss += loss.item()

        avg_loss = total_loss / len(labeled_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def train_with_pseudo_and_mean_teacher(gnn, labeled_graphs, labeled_loader, unlabeled_loader, epochs=50, alpha=0.99, consistency_weight=1.0):
    """
    2+3+4: supervised CE + pseudo-labels (from teacher) every 5 epochs + Mean-Teacher consistency.
    Pseudo-labels are generated from the teacher (stable EMA model) rather than the noisy student.
    """
    teacher = copy.deepcopy(gnn)
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(gnn.parameters(), lr=0.01)
    current_labeled_loader = labeled_loader
    current_labeled_graphs = labeled_graphs
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(epochs):
        # regenerate pseudo-labels from teacher every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            pseudo_graphs = generate_pseudo_labels(teacher, unlabeled_loader)
            current_labeled_graphs = labeled_graphs + pseudo_graphs
            current_labeled_loader = DataLoader(current_labeled_graphs, batch_size=32, shuffle=True)

        gnn.train()
        teacher.train()
        total_loss = 0

        for labeled_batch in current_labeled_loader:
            optimizer.zero_grad()

            # supervised loss on labeled + pseudo-labeled batch
            student_out = gnn(labeled_batch.x, labeled_batch.edge_index, labeled_batch.batch)
            sup_loss = criterion(student_out, labeled_batch.y)

            # consistency loss on unlabeled batch
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            student_unlabeled = gnn(unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.batch)
            with torch.no_grad():
                teacher_unlabeled = teacher(unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_batch.batch)

            consistency_loss = F.mse_loss(
                torch.softmax(student_unlabeled, dim=1),
                torch.softmax(teacher_unlabeled, dim=1)
            )

            loss = sup_loss + consistency_weight * consistency_loss
            loss.backward()
            optimizer.step()
            update_ema(gnn, teacher, alpha=alpha)
            total_loss += loss.item()

        avg_loss = total_loss / len(current_labeled_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
    return correct / total if total > 0 else 0.0
