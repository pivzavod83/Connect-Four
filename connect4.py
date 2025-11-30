"""
Connect-4 Game

Task implementations:
- Task 1: Lines ... (Game setup, board, user input, display, win checking)
          Functions: __init__,_on_gui_column_click, display_board, is_valid_move, get_valid_moves,
          make_move, undo_move, check_winner, is_board_full, play, main, class BoardDisplay
- Task 2: Lines ..., ... (Legal moves generation, search space display)
- Task 3: Lines ... (Minimax algorithm implementation)
- Task 4: Lines ... (Alpha-beta pruning integrated in minimax)

To RUN:
conda env create -f environment.yaml
conda activate connect4
python connect4.py
"""

import numpy as np
from typing import List, Tuple, Optional
import threading
import time
# Import GUI display
from board_display import BoardDisplay


class Connect4:
    
    EMPTY = 0
    HUMAN = 1  # Red
    COMPUTER = 2  # Yellow
    
    def __init__(self, rows=6, cols=7):
        # Initialize the game board
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int) # init the board as a 2D array of zeros
        self.current_player = self.HUMAN  # human goes first
        self.game_over = False
        self.winner = None
        
        # GUI setup
        self.gui_display = BoardDisplay(rows, cols, on_column_click=self._on_gui_column_click)
        self.gui_display.update_board(self.board, self.HUMAN, self.COMPUTER)
        self.selected_column = None
        self.waiting_for_input = False
    
    def _on_gui_column_click(self, col: int):
        # Handle column click from GUI
        if self.waiting_for_input and self.is_valid_move(col): # if my turn and move is valid
            self.selected_column = col # save the column
            self.waiting_for_input = False
        
    def display_board(self):
        # Display the CURRENT board state
        self.gui_display.update_board(self.board, self.HUMAN, self.COMPUTER)
        self.gui_display.update()
        
    def is_valid_move(self, col: int) -> bool:
        # Check if a move is valid (column not full)
        if col < 0 or col >= self.cols:
            return False # out of bounds
        return self.board[0][col] == self.EMPTY # board[0][col] is the top of the column, check if it's empty
    
    def get_valid_moves(self) -> List[int]:
        # Get all valid moves
        valid_moves = [] # list of valid moves
        for col in range(self.cols):
            if self.is_valid_move(col): # if the column is valid
                valid_moves.append(col)
        return valid_moves
    
    def make_move(self, col: int, player: int) -> bool:
        # Drop a disc

        if not self.is_valid_move(col):
            return False
        
        # Find the lowest available row in the column
        for row in range(self.rows - 1, -1, -1): # start from the bottom of the column
            if self.board[row][col] == self.EMPTY:
                self.board[row][col] = player # drop the disc
                return True
        return False # column is full
    
    def undo_move(self, col: int):
        # Remove the top disc from the column
        for row in range(self.rows): # start from the top of the column
            if self.board[row][col] != self.EMPTY:
                self.board[row][col] = self.EMPTY # remove the disc
                return
    
    def check_winner(self) -> Optional[int]:
        # Check for a winner
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal (top-left to bottom-right)
            (1, -1)   # Diagonal (top-right to bottom-left)
        ]
        
        for row in range(self.rows):
            for col in range(self.cols): # iterate over all slots
                if self.board[row][col] == self.EMPTY: # if the slot is empty, ignore it
                    continue
                
                player = self.board[row][col] # get the player
                
                for dr, dc in directions: # iterate over all directions
                    count = 1 # count the number of discs in a row
                    for i in range(1, 4): # iterate over the next 3 slots
                        new_row = row + dr * i
                        new_col = col + dc * i
                        
                        if (0 <= new_row < self.rows and # check if the new row is in bounds
                            0 <= new_col < self.cols and # check if the new column is in bounds
                            self.board[new_row][new_col] == player):
                            count += 1 # increment the count
                        else:
                            break
                    
                    if count >= 4:
                        return player
        return None # no winner
    
    def is_board_full(self) -> bool:
        # Check if the board is completely full
        return all(self.board[0][col] != self.EMPTY for col in range(self.cols))
    
    def evaluate_position(self) -> int:
        # Evaluate the current board position from computer's perspective.
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
        # Generate search space information for display.
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
        # Minimax algorithm with optional alpha-beta pruning
        # Returns (best_score, stats_dict)
        
        if stats is None:
            stats = {
                'nodes_evaluated': 0,
                'nodes_pruned': 0,
                'depth_reached': {},
                'scores_at_depth': {}
            }
        
        stats['nodes_evaluated'] += 1
        
        # Check for terminal states - WIN CHECK FIRST (most important!)
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
        # Get the best move for the computer using minimax
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return -1, {}
        
        # FIRST: Check for immediate winning moves (highest priority!)
        for move in valid_moves:
            self.make_move(move, self.COMPUTER)
            if self.check_winner() == self.COMPUTER:
                self.undo_move(move)
                # Return immediately - this is a winning move!
                return move, {
                    'nodes_evaluated': 1,
                    'nodes_pruned': 0,
                    'depth_reached': {0: 1},
                    'scores_at_depth': {0: [1000]}
                }
            self.undo_move(move)
        
        # SECOND: Check if we need to block human from winning
        for move in valid_moves:
            self.make_move(move, self.HUMAN)
            if self.check_winner() == self.HUMAN:
                self.undo_move(move)
                # Block this move - human would win otherwise
                return move, {
                    'nodes_evaluated': 1,
                    'nodes_pruned': 0,
                    'depth_reached': {0: 1},
                    'scores_at_depth': {0: [-1000]}
                }
            self.undo_move(move)
        
        # THIRD: Use minimax to find best move
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
        # Recursively calculate the actual search space at a given level
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
        # Display search space information for all levels
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
        # Display minimax algorithm statistics
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
        # Main game loop
        # Print the game setup
        print("\n" + "=" * 50)
        print("CONNECT-4 GAME")
        print("=" * 50)
        print("Human plays RED, Computer plays YELLOW")
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
                if not valid_moves: # no valid moves
                    self.gui_display.show_draw()
                    print("No valid moves available. Game is a draw!")
                    break
                
                # GUI for input
                self.gui_display.set_status("Click a column to drop a disc")
                self.gui_display.enable_buttons(valid_moves) # enable the buttons for the valid moves
                self.waiting_for_input = True
                self.selected_column = None
                
                # Wait for user to click a column
                while self.waiting_for_input and not self.game_over:
                    time.sleep(0.1) # wait for 0.1 seconds
                    self.gui_display.update() # update the GUI
                
                if self.selected_column is not None: # if a column is selected
                    col = self.selected_column
                    # Find the row where the disc will land
                    for row in range(self.rows - 1, -1, -1): # start from the bottom of the column
                        if self.board[row][col] == self.EMPTY: # if the slot is empty
                            self.make_move(col, self.HUMAN) # make the move
                            self.gui_display.animate_drop(col, row, self.HUMAN, self.board) # animate the drop
                            break
                
            else:
                # Computer's turn
                self.gui_display.set_status("Computer is thinking...")
                self.gui_display.disable_buttons()
                
                print("\nComputer's turn (YELLOW)...")
                best_move, stats = self.get_best_move(search_depth, use_alpha_beta)
                
                # Display minimax statistics
                self.display_minimax_stats(stats, use_alpha_beta)
                
                if best_move == -1: # no valid moves
                    self.gui_display.show_draw() # show the draw message
                    print("No valid moves available, game is a draw!")
                    break
                
                # Drop the disc
                for row in range(self.rows - 1, -1, -1):
                    if self.board[row][best_move] == self.EMPTY:
                        self.make_move(best_move, self.COMPUTER) 
                        self.gui_display.animate_drop(best_move, row, self.COMPUTER, self.board)
                        break
                
                print(f"Computer plays column {best_move}")
            
            # Check for winner
            winner = self.check_winner()
            if winner:
                self.game_over = True # game is over
                self.winner = winner
                self.display_board() # display the board
                self.gui_display.show_winner(winner, winner == self.HUMAN) # show the winner message
                if winner == self.HUMAN: # if the human won
                    print("\nYou won!")
                else:
                    print("\nComputer wins!")
                break
            
            # Check for draw
            if self.is_board_full():
                self.game_over = True # game is over
                self.display_board()
                self.gui_display.show_draw()
                print("\nIt's a draw!")
                break
            
            # Switch players
            self.current_player = self.COMPUTER if self.current_player == self.HUMAN else self.HUMAN
        
        print("\nGame Over!")
        
        # Keep GUI running
        self.gui_display.set_status("Game Over! Close the window to exit.")
        # Update GUI periodically
        try:
            for _ in range(300):  # wait up to 30 seconds
                self.gui_display.update() # update the GUI
                time.sleep(0.1) # wait for 0.1 seconds
        except:
            pass # if error, ignore it

def main():
    print("Connect-4 Game - AICE 2002 Assignment 2")
    print("\nConfiguration:")
    
    try: # Try to get input from the user
        depth = int(input("Enter search depth (recommended: 3-5): ") or "3")
        use_ab = input("Use alpha-beta pruning? (y/n): ").lower() != 'n'
    except (ValueError, EOFError): # If fails, use default values
        depth = 3
        use_ab = True
        print("Using default values: depth=3, alpha-beta=True") # Default values
    
    game = Connect4() # Create a game instance
    
    # Run game with GUI
    import threading
    game_thread = threading.Thread(target=game.play, args=(depth, use_ab), daemon=True) # create a thread for the game
    game_thread.start()
    
    # Run GUI in main thread
    game.gui_display.mainloop()


if __name__ == "__main__":
    main()

