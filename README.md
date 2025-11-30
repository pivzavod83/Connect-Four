# Connect-4 Game with AI

An intelligent Connect-4 game implementation featuring a GUI interface and AI opponent using Minimax algorithm with Alpha-Beta pruning. *Built as part of AICE 2002 - AI & Learning Machines Assignment 2.*

## 🎮 Features

- **Interactive GUI**: Modern, user-friendly graphical interface built with Tkinter
- **AI Opponent**: Computer player powered by Minimax algorithm
- **Alpha-Beta Pruning**: Optimized search with configurable pruning
- **Search Space Analysis**: Displays search space information at each level
- **Game Statistics**: Detailed minimax algorithm statistics and performance metrics

## 📋 Requirements

- Python 3.9+
- NumPy >= 1.21.0
- Tkinter

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/pivzavod83/Connect-Four.git
cd Connect-Four
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate connect4
```

Alternatively, if you prefer pip:
```bash
pip install numpy
```

## 🎯 Usage

Run the game:
```bash
python connect4.py
```

### How to Play

1. Click on a column button (1-7) to drop your disc
2. The computer will automatically make its move
3. First player to get 4 in a row wins!


## 🧠 Algorithm Details

### Minimax Algorithm
The AI uses the Minimax algorithm to evaluate all possible moves up to a specified depth, choosing the move that maximizes its chances of winning while minimizing the opponent's chances.

### Alpha-Beta Pruning
Alpha-beta pruning significantly reduces the number of nodes evaluated by eliminating branches that cannot possibly influence the final decision, improving performance without affecting the optimality of the solution.

## 📝 License

See LICENSE file for details.