import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim = 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Data loading
category_lines, all_categories = load_data()
# Dict: country:names, List of countries 

n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# One step for RNN
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
# print(output.shape)
# print(next_hidden.shape)

# Whole sequence
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
# print(output.shape)
# print(next_hidden.shape)

def category_from_output(output):
    return all_categories[torch.argmax(output).item()]


criterion = nn.NLLLoss()
lr = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr = lr)

writer = SummaryWriter('runs vs loss')
steps = 0

def train(line_tensor, category_tensor):
    
    hidden = rnn.init_hidden()

    for i in range(line_tensor.shape[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    global steps
    writer.add_scalar('Training loss', loss, global_step = steps)
    steps += 1

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000

n_iters = 100000

for i in tqdm(range(n_iters)):
    category, line, category_tensor, line_tensor = random_training_example(category_lines=category_lines, all_categories = all_categories)

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
    
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess==category else f"WRONG: should be {category}"
        print(f"{i} {(i+1)* 100//n_iters }% {loss:.4f} {line} / {guess} {correct}")


def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.shape[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        print(guess)

while True:
    sentence = input("Input: ")
    if sentence == "quit":
        break

    predict(sentence)