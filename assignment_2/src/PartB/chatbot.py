import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path


"""Use a pretrained Transformer model to build a simple movie chatbot based on IMDb-style text data.
You should be able to ask questions such as:
"Who is the villain in Star Wars?"
 "What movie features Darth Vader?"

The goal is code that can work as follows:
>>> question = "Who is the villain in Star Wars?"
>>> print(model.generate(question))
“Darth Vader”

The model does not need to be trained to converge.
"""


class IMDBDataset(Dataset):
    """Dataset that creates Q&A pairs from IMDB movie data"""
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qa_pairs = self._create_qa_pairs()
    
    def _create_qa_pairs(self):
        """Create question-answer pairs from movie data"""
        qa_pairs = []
        
        for _, row in self.df.iterrows():
            title = str(row['Series_Title'])
            director = str(row['Director']) if pd.notna(row['Director']) else "Unknown"
            star1 = str(row['Star1']) if pd.notna(row['Star1']) else "Unknown"
            star2 = str(row['Star2']) if pd.notna(row['Star2']) else "Unknown"
            genre = str(row['Genre']) if pd.notna(row['Genre']) else "Unknown"
            overview = str(row['Overview']) if pd.notna(row['Overview']) else "Unknown"
            year = str(row['Released_Year']) if pd.notna(row['Released_Year']) else "Unknown"
            
            # Create various Q&A pairs
            qa_pairs.append({
                'question': f"Who directed {title}?",
                'answer': director
            })
            qa_pairs.append({
                'question': f"Who stars in {title}?",
                'answer': f"{star1} and {star2}"
            })
            qa_pairs.append({
                'question': f"What is {title} about?",
                'answer': overview
            })
            qa_pairs.append({
                'question': f"What genre is {title}?",
                'answer': genre
            })
            qa_pairs.append({
                'question': f"When was {title} released?",
                'answer': year
            })
            
        return qa_pairs
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        
        # Tokenize question
        input_encoding = self.tokenizer(
            pair['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer
        target_encoding = self.tokenizer(
            pair['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }


class SimpleChatbot(nn.Module):
    def __init__(self, transformer_model):
        super(SimpleChatbot, self).__init__()
        self.transformer = transformer_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def generate(self, question, tokenizer, max_length=50):
        inputs = tokenizer.encode(question, return_tensors='pt')
        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        outputs = self.transformer.generate(inputs, max_length=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def train_chatbot(model, train_loader, tokenizer, epochs=3, lr=5e-5):
    """Train the chatbot on IMDB data"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    print(f"Training on {device}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # For seq2seq models, pass labels and let the model handle loss computation
            outputs = model.transformer(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")
    
    return model
    

if __name__ == "__main__":
    # Options for better chatbot models:
    # 1. "google/flan-t5-small" - T5 fine-tuned for instructions/QA
    # 2. "facebook/blenderbot-400M-distill" - Conversational model
    # 3. "google/flan-t5-base" - Larger, better instruction following
    
    model_name = "google/flan-t5-small"  # Better for Q&A tasks
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    chatbot = SimpleChatbot(transformer_model)
    
    # Load IMDB dataset
    data_path = Path(__file__).parent.parent.parent / "data" / "imdb" / "imdb_top_1000.csv"
    print(f"Loading dataset from: {data_path}")
    
    dataset = IMDBDataset(data_path, tokenizer, max_length=128)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} Q&A pairs")
    
    # Train the model (just a few epochs since it doesn't need to converge)
    print("\nTraining chatbot on IMDB data...")
    chatbot = train_chatbot(chatbot, train_loader, tokenizer, epochs=2, lr=5e-5)
    
    # Test the chatbot
    print("\n" + "="*50)
    print("Testing chatbot:")
    print("="*50)
    
    test_questions = [
        "Who directed The Dark Knight?",
        "What is Inception about?",
        "Who stars in The Godfather?",
        "What genre is Pulp Fiction?"
    ]
    
    chatbot.eval()
    for question in test_questions:
        answer = chatbot.generate(question, tokenizer)
        print(f"\nQ: {question}")
        print(f"A: {answer}")    