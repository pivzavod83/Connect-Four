"""
Connect-4 Game Implementation

Task Implementation:
- Task 1: Lines 11-103 (Game setup, board, user input, display, win checking)
- Task 2: Lines 165-186, 310-353 (Legal moves generation, search space display)
- Task 3: Lines 188-285 (Minimax algorithm implementation)
- Task 4: Lines 188-285 (Alpha-beta pruning integrated in minimax)
"""

import numpy as np
from typing import List, Tuple, Optional


class Connect4:
    """Connect-4 game implementation with minimax and alpha-beta pruning."""
    
    EMPTY = 0
    HUMAN = 1  # Red
    COMPUTER = 2  # Yellow
    
    def __init__(self, rows=6, cols=7):
        """Initialize the game board."""
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = self.HUMAN  # Human goes first
        self.game_over = False
        self.winner = None
        
    def display_board(self):
        """Display the current board state."""
        print("\n" + "=" * 50)
        print("Current Board State:")
        print("=" * 50)
        # Print column numbers
        print("  ", end="")
        for col in range(self.cols):
            print(f" {col} ", end="")
        print()
        
        # Print board with row numbers
        for row in range(self.rows):
            print(f"{row} ", end="")
            for col in range(self.cols):
                if self.board[row][col] == self.EMPTY:
                    print(" . ", end="")
                elif self.board[row][col] == self.HUMAN:
                    print(" R ", end="")  # Red
                else:
                    print(" Y ", end="")  # Yellow
            print()
        print("=" * 50)
        
    def is_valid_move(self, col: int) -> bool:
        """Check if a move is valid (column not full)."""
        if col < 0 or col >= self.cols:
            return False
        return self.board[0][col] == self.EMPTY
    
    def get_valid_moves(self) -> List[int]:
        """Get all valid moves (columns that are not full)."""
        valid_moves = []
        for col in range(self.cols):
            if self.is_valid_move(col):
                valid_moves.append(col)
        return valid_moves
    
    def make_move(self, col: int, player: int) -> bool:
        """Make a move by dropping a disc in the specified column."""
        if not self.is_valid_move(col):
            return False
        
        # Find the lowest available row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == self.EMPTY:
                self.board[row][col] = player
                return True
        return False
    
    def undo_move(self, col: int):
        """Undo a move by removing the top disc from the column."""
        for row in range(self.rows):
            if self.board[row][col] != self.EMPTY:
                self.board[row][col] = self.EMPTY
                return
    
    def check_winner(self) -> Optional[int]:
        """Check if there's a winner. Returns player number or None."""
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal (top-left to bottom-right)
            (1, -1)   # Diagonal (top-right to bottom-left)
        ]
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == self.EMPTY:
                    continue
                
                player = self.board[row][col]
                
                for dr, dc in directions:
                    count = 1
                    for i in range(1, 4):
                        new_row = row + dr * i
                        new_col = col + dc * i
                        
                        if (0 <= new_row < self.rows and 
                            0 <= new_col < self.cols and 
                            self.board[new_row][new_col] == player):
                            count += 1
                        else:
                            break
                    
                    if count >= 4:
                        return player
        
        return None
    
    def is_board_full(self) -> bool:
        """Check if the board is completely full."""
        return all(self.board[0][col] != self.EMPTY for col in range(self.cols))
    
    def evaluate_position(self) -> int:
        """Evaluate the current board position from computer's perspective."""
        winner = self.check_winner()
        if winner == self.COMPUTER:
            return 1000  # Computer wins
        elif winner == self.HUMAN:
            return -1000  # Human wins
        
        if self.is_board_full():
            return 0  # Draw
        
        # Heuristic: count potential winning lines
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(self.rows):
            for col in range(self.cols):
                for dr, dc in directions:
                    computer_count = 0
                    human_count = 0
                    empty_count = 0
                    
                    for i in range(4):
                        new_row = row + dr * i
                        new_col = col + dc * i
                        
                        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
                            break
                        
                        if self.board[new_row][new_col] == self.COMPUTER:
                            computer_count += 1
                        elif self.board[new_row][new_col] == self.HUMAN:
                            human_count += 1
                        else:
                            empty_count += 1
                    else:
                        # All 4 positions are valid
                        if computer_count > 0 and human_count == 0:
                            score += computer_count ** 2
                        elif human_count > 0 and computer_count == 0:
                            score -= human_count ** 2
        
        return score
    
    def generate_search_space_info(self, depth: int, current_depth: int = 0) -> dict:
        """Generate search space information for display."""
        info = {
            'level': current_depth,
            'moves': len(self.get_valid_moves()),
            'total_moves': 0
        }
        
        if current_depth < depth and not self.check_winner() and not self.is_board_full():
            valid_moves = self.get_valid_moves()
            total = 0
            for move in valid_moves:
                self.make_move(move, self.HUMAN if current_depth % 2 == 0 else self.COMPUTER)
                sub_info = self.generate_search_space_info(depth, current_depth + 1)
                total += sub_info['total_moves'] if sub_info['total_moves'] > 0 else 1
                self.undo_move(move)
            
            info['total_moves'] = len(valid_moves) * (total if total > 0 else 1)
        else:
            info['total_moves'] = 1
        
        return info
    
    def minimax(self, depth: int, is_maximizing: bool, 
                alpha: float = float('-inf'), 
                beta: float = float('inf'),
                use_alpha_beta: bool = False,
                stats: dict = None) -> Tuple[int, dict]:
        """
        Minimax algorithm with optional alpha-beta pruning.
        Returns (best_score, stats_dict)
        """
        if stats is None:
            stats = {
                'nodes_evaluated': 0,
                'nodes_pruned': 0,
                'depth_reached': {},
                'scores_at_depth': {}
            }
        
        stats['nodes_evaluated'] += 1
        
        # Check for terminal states
        winner = self.check_winner()
        if winner == self.COMPUTER:
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(1000)
            return 1000, stats
        elif winner == self.HUMAN:
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(-1000)
            return -1000, stats
        
        if self.is_board_full():
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(0)
            return 0, stats
        
        if depth == 0:
            score = self.evaluate_position()
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(score)
            return score, stats
        
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            score = self.evaluate_position()
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(score)
            return score, stats
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                self.make_move(move, self.COMPUTER)
                eval_score, stats = self.minimax(depth - 1, False, alpha, beta, 
                                                 use_alpha_beta, stats)
                self.undo_move(move)
                
                max_eval = max(max_eval, eval_score)
                
                if use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        stats['nodes_pruned'] += len(valid_moves) - valid_moves.index(move) - 1
                        break
            
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(max_eval)
            return max_eval, stats
        else:
            min_eval = float('inf')
            for move in valid_moves:
                self.make_move(move, self.HUMAN)
                eval_score, stats = self.minimax(depth - 1, True, alpha, beta, 
                                                 use_alpha_beta, stats)
                self.undo_move(move)
                
                min_eval = min(min_eval, eval_score)
                
                if use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        stats['nodes_pruned'] += len(valid_moves) - valid_moves.index(move) - 1
                        break
            
            stats['depth_reached'][depth] = stats['depth_reached'].get(depth, 0) + 1
            if depth not in stats['scores_at_depth']:
                stats['scores_at_depth'][depth] = []
            stats['scores_at_depth'][depth].append(min_eval)
            return min_eval, stats
    
    def get_best_move(self, depth: int, use_alpha_beta: bool = False) -> Tuple[int, dict]:
        """Get the best move for the computer using minimax."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return -1, {}
        
        best_move = valid_moves[0]
        best_score = float('-inf')
        stats = {
            'nodes_evaluated': 0,
            'nodes_pruned': 0,
            'depth_reached': {},
            'scores_at_depth': {}
        }
        
        for move in valid_moves:
            self.make_move(move, self.COMPUTER)
            score, stats = self.minimax(depth - 1, False, float('-inf'), float('inf'), 
                                       use_alpha_beta, stats)
            self.undo_move(move)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move, stats
    
    def calculate_search_space_at_level(self, level: int, max_depth: int, player: int, max_calc_depth: int = 3) -> int:
        """Recursively calculate the actual search space at a given level."""
        # Limit calculation depth to avoid long waits
        if level >= max_depth or level >= max_calc_depth:
            valid_moves = self.get_valid_moves()
            return len(valid_moves) if valid_moves else 1
        
        if self.check_winner() or self.is_board_full():
            return 1
        
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return 1
        
        total = 0
        next_player = self.COMPUTER if player == self.HUMAN else self.HUMAN
        
        for move in valid_moves:
            self.make_move(move, player)
            total += self.calculate_search_space_at_level(level + 1, max_depth, next_player, max_calc_depth)
            self.undo_move(move)
        
        return total
    
    def display_search_space_info(self, depth: int):
        """Display search space information for all levels."""
        print("\n" + "=" * 50)
        print("SEARCH SPACE INFORMATION")
        print("=" * 50)
        print("Calculating search space for each level...")
        
        current_player = self.current_player
        
        for level in range(depth + 1):
            if level == 0:
                player_name = "Current Turn (Root)"
                player = current_player
            elif level % 2 == 1:
                player_name = "Human Turn" if current_player == self.HUMAN else "Computer Turn"
                player = self.HUMAN if current_player == self.HUMAN else self.COMPUTER
            else:
                player_name = "Computer Turn" if current_player == self.HUMAN else "Human Turn"
                player = self.COMPUTER if current_player == self.HUMAN else self.HUMAN
            
            # Get valid moves at this level
            valid_moves = self.get_valid_moves()
            num_moves = len(valid_moves)
            
            # Calculate actual search space if not too deep
            if level == 0:
                total_at_level = num_moves
                print(f"Level {level} ({player_name}): {num_moves} possible moves")
            else:
                # For deeper levels, calculate recursively (but limit to avoid long waits)
                # Calculate up to 3 levels deep for accuracy, then estimate
                if level <= 3:
                    total_at_level = self.calculate_search_space_at_level(level, depth, player, max_calc_depth=3)
                    print(f"Level {level} ({player_name}): {num_moves} possible moves")
                    if level < depth:
                        print(f"  Calculated positions at this level: {total_at_level}")
                        if depth > 3:
                            estimated = num_moves * (7 ** (depth - level))
                            print(f"  Estimated total positions (full depth): ~{estimated}")
                else:
                    # For deeper searches, provide estimate
                    total_at_level = num_moves * (7 ** (depth - level))
                    print(f"Level {level} ({player_name}): {num_moves} possible moves")
                    print(f"  Estimated total positions at this level: ~{total_at_level}")
        
        print("=" * 50)
    
    def display_minimax_stats(self, stats: dict, use_alpha_beta: bool = False):
        """Display minimax algorithm statistics."""
        print("\n" + "=" * 50)
        if use_alpha_beta:
            print("MINIMAX WITH ALPHA-BETA PRUNING RESULTS")
        else:
            print("MINIMAX RESULTS")
        print("=" * 50)
        print(f"Total nodes evaluated: {stats['nodes_evaluated']}")
        if use_alpha_beta:
            print(f"Total nodes pruned: {stats['nodes_pruned']}")
            print(f"Pruning efficiency: {stats['nodes_pruned'] / max(stats['nodes_evaluated'], 1) * 100:.2f}%")
        
        print("\nDepth exploration summary:")
        for depth in sorted(stats['depth_reached'].keys(), reverse=True):
            count = stats['depth_reached'][depth]
            print(f"  Depth {depth}: {count} nodes evaluated")
        
        print("\nScore distribution by depth:")
        for depth in sorted(stats['scores_at_depth'].keys(), reverse=True):
            scores = stats['scores_at_depth'][depth]
            if scores:
                print(f"  Depth {depth}: min={min(scores)}, max={max(scores)}, "
                      f"avg={sum(scores)/len(scores):.2f}, count={len(scores)}")
        
        print("=" * 50)
    
    def play(self, search_depth: int = 3, use_alpha_beta: bool = False):
        """Main game loop."""
        print("\n" + "=" * 50)
        print("CONNECT-4 GAME")
        print("=" * 50)
        print("Human plays RED (R), Computer plays YELLOW (Y)")
        print(f"Human goes first!")
        print(f"Search depth: {search_depth}")
        print(f"Alpha-beta pruning: {'Enabled' if use_alpha_beta else 'Disabled'}")
        print("=" * 50)
        
        # Display initial search space info
        self.display_search_space_info(search_depth)
        
        while not self.game_over:
            self.display_board()
            
            if self.current_player == self.HUMAN:
                # Human's turn
                valid_moves = self.get_valid_moves()
                if not valid_moves:
                    print("No valid moves available. Game is a draw!")
                    break
                
                print(f"\nYour turn (RED). Valid columns: {valid_moves}")
                try:
                    col = int(input("Enter column number (0-6): "))
                    if col not in valid_moves:
                        print("Invalid move! Please choose a valid column.")
                        continue
                except ValueError:
                    print("Invalid input! Please enter a number.")
                    continue
                
                self.make_move(col, self.HUMAN)
                
            else:
                # Computer's turn
                print("\nComputer's turn (YELLOW)...")
                best_move, stats = self.get_best_move(search_depth, use_alpha_beta)
                
                # Display minimax statistics
                self.display_minimax_stats(stats, use_alpha_beta)
                
                if best_move == -1:
                    print("No valid moves available. Game is a draw!")
                    break
                
                self.make_move(best_move, self.COMPUTER)
                print(f"Computer plays column {best_move}")
            
            # Check for winner
            winner = self.check_winner()
            if winner:
                self.game_over = True
                self.winner = winner
                self.display_board()
                if winner == self.HUMAN:
                    print("\n🎉 Congratulations! You won!")
                else:
                    print("\n💻 Computer wins!")
                break
            
            # Check for draw
            if self.is_board_full():
                self.game_over = True
                self.display_board()
                print("\n🤝 It's a draw!")
                break
            
            # Switch players
            self.current_player = self.COMPUTER if self.current_player == self.HUMAN else self.HUMAN
        
        print("\nGame Over!")


def main():
    """Main function to run the game."""
    print("Connect-4 Game - AICE 2002 Assignment 2")
    print("\nConfiguration:")
    
    try:
        depth = int(input("Enter search depth (recommended: 3-5): ") or "3")
        use_ab = input("Use alpha-beta pruning? (y/n, default: y): ").lower() != 'n'
    except ValueError:
        depth = 3
        use_ab = True
        print("Using default values: depth=3, alpha-beta=True")
    
    game = Connect4()
    game.play(search_depth=depth, use_alpha_beta=use_ab)


if __name__ == "__main__":
    main()

