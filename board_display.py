"""
GUI Display Module for Connect-4 Game
Creates a colorful pop-up window to display the game board
"""

import tkinter as tk
from tkinter import messagebox
from typing import Optional, Callable
import numpy as np


class BoardDisplay:
    """GUI display for Connect-4 board with colorful visualization."""
    
    def __init__(self, rows=6, cols=7, on_column_click: Optional[Callable[[int], None]] = None):
        """
        Initialize the GUI board display.
        
        Args:
            rows: Number of rows in the board
            cols: Number of columns in the board
            on_column_click: Callback function when a column is clicked (takes column index)
        """
        self.rows = rows
        self.cols = cols
        self.on_column_click = on_column_click
        
        # Colors
        self.BOARD_BG = "#1E3A8A"  # Dark blue background
        self.EMPTY_COLOR = "#FFFFFF"  # White for empty slots
        self.HUMAN_COLOR = "#EF4444"  # Red for human player
        self.COMPUTER_COLOR = "#FCD34D"  # Yellow for computer player
        self.SLOT_OUTLINE = "#1E40AF"  # Blue outline for slots
        self.BUTTON_BG = "#3B82F6"  # Blue for column buttons
        self.BUTTON_HOVER = "#2563EB"  # Darker blue on hover
        self.TEXT_COLOR = "#FFFFFF"  # White text
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Connect-4 Game")
        self.root.configure(bg="#0F172A")  # Dark background
        
        # Calculate dimensions
        self.slot_size = 70
        self.slot_padding = 5
        self.board_width = cols * (self.slot_size + self.slot_padding) + self.slot_padding
        self.board_height = rows * (self.slot_size + self.slot_padding) + self.slot_padding
        self.button_height = 40
        
        # Window size - buttons are now under columns, so board width is sufficient
        window_width = self.board_width + 40
        window_height = self.board_height + self.button_height + 140  # Extra space for buttons under board
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        
        # Create a frame that contains both board and buttons aligned
        main_frame = tk.Frame(self.root, bg="#0F172A")
        main_frame.pack(pady=20)
        
        # Create canvas for board
        self.canvas = tk.Canvas(
            main_frame,
            width=self.board_width,
            height=self.board_height,
            bg=self.BOARD_BG,
            highlightthickness=0
        )
        self.canvas.pack(pady=(0, 10))
        
        # Create column buttons frame positioned directly under the board
        self.button_frame = tk.Frame(main_frame, bg="#0F172A")
        self.button_frame.pack(pady=10)
        
        self.buttons = []
        # Simple, normal buttons
        for col in range(cols):
            btn = tk.Button(
                self.button_frame,
                text=str(col + 1),  # Display 1-7 instead of 0-6
                width=6,
                height=2,
                bg=self.BUTTON_BG,
                fg=self.TEXT_COLOR,
                font=("Arial", 11, "bold"),
                relief="raised",
                bd=3,
                command=lambda c=col: self._on_button_click(c),
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=3)
            self.buttons.append(btn)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Your turn! Click a column to drop a disc.",
            bg="#0F172A",
            fg=self.TEXT_COLOR,
            font=("Arial", 14, "bold"),
            pady=10
        )
        self.status_label.pack()
        
        # Store current board state
        self.current_board = np.zeros((rows, cols), dtype=int)
        self.human_color = 1
        self.computer_color = 2
        
        # Draw initial empty board
        self._draw_board()
        
        # Bind hover effects
        self._bind_hover_effects()
    
    def _bind_hover_effects(self):
        """Add hover effects to buttons."""
        for i, btn in enumerate(self.buttons):
            def on_enter(event, button=btn):
                button.config(bg=self.BUTTON_HOVER)
            
            def on_leave(event, button=btn):
                button.config(bg=self.BUTTON_BG)
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
    
    def _on_button_click(self, col: int):
        """Handle column button click."""
        if self.on_column_click:
            self.on_column_click(col)
    
    def _draw_board(self):
        """Draw the game board on the canvas."""
        self.canvas.delete("all")
        
        # Draw board background (dark blue)
        self.canvas.create_rectangle(
            0, 0, self.board_width, self.board_height,
            fill=self.BOARD_BG, outline=""
        )
        
        # Draw slots (circles)
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * (self.slot_size + self.slot_padding) + self.slot_padding + self.slot_size // 2
                y = row * (self.slot_size + self.slot_padding) + self.slot_padding + self.slot_size // 2
                
                # Determine color based on board state
                if self.current_board[row][col] == self.human_color:
                    color = self.HUMAN_COLOR
                elif self.current_board[row][col] == self.computer_color:
                    color = self.COMPUTER_COLOR
                else:
                    color = self.EMPTY_COLOR
                
                # Draw circle (disc or empty slot)
                radius = self.slot_size // 2 - 2
                self.canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color,
                    outline=self.SLOT_OUTLINE,
                    width=2
                )
    
    def update_board(self, board: np.ndarray, human_color: int = 1, computer_color: int = 2):
        """
        Update the board display with new board state.
        
        Args:
            board: 2D numpy array representing the board
            human_color: Value representing human player (default: 1)
            computer_color: Value representing computer player (default: 2)
        """
        self.current_board = board.copy()
        self.human_color = human_color
        self.computer_color = computer_color
        self._draw_board()
        self.root.update()
    
    def set_status(self, message: str):
        """Update the status message."""
        self.status_label.config(text=message)
        self.root.update()
    
    def enable_buttons(self, valid_columns: Optional[list] = None):
        """
        Enable column buttons. If valid_columns is provided, only enable those.
        
        Args:
            valid_columns: List of valid column indices (None means all columns)
        """
        for i, btn in enumerate(self.buttons):
            if valid_columns is None or i in valid_columns:
                btn.config(state=tk.NORMAL, bg=self.BUTTON_BG)
            else:
                btn.config(state=tk.DISABLED, bg="#64748B")
    
    def disable_buttons(self):
        """Disable all column buttons."""
        for btn in self.buttons:
            btn.config(state=tk.DISABLED, bg="#64748B")
    
    def show_winner(self, winner: int, is_human: bool):
        """
        Show winner message.
        
        Args:
            winner: Winner player value
            is_human: True if human won, False if computer won
        """
        self.disable_buttons()
        if is_human:
            message = "🎉 Congratulations! You won!"
            color = self.HUMAN_COLOR
        else:
            message = "💻 Computer wins!"
            color = self.COMPUTER_COLOR
        
        self.set_status(message)
        messagebox.showinfo("Game Over", message)
    
    def show_draw(self):
        """Show draw message."""
        self.disable_buttons()
        message = "🤝 It's a draw!"
        self.set_status(message)
        messagebox.showinfo("Game Over", message)
    
    def animate_drop(self, col: int, row: int, player: int, board: np.ndarray):
        """
        Animate a disc dropping into a column.
        
        Args:
            col: Column index
            row: Final row position
            player: Player value (1 for human, 2 for computer)
            board: Updated board state
        """
        # Animate disc falling
        start_y = -self.slot_size
        end_y = row * (self.slot_size + self.slot_padding) + self.slot_padding + self.slot_size // 2
        x = col * (self.slot_size + self.slot_padding) + self.slot_padding + self.slot_size // 2
        
        color = self.HUMAN_COLOR if player == self.human_color else self.COMPUTER_COLOR
        radius = self.slot_size // 2 - 2
        
        # Create animated disc
        disc = self.canvas.create_oval(
            x - radius, start_y - radius,
            x + radius, start_y + radius,
            fill=color,
            outline=self.SLOT_OUTLINE,
            width=2
        )
        
        # Animate falling
        steps = 20
        step_size = (end_y - start_y) / steps
        
        def animate(step=0):
            if step < steps:
                y_pos = start_y + step * step_size
                self.canvas.coords(
                    disc,
                    x - radius, y_pos - radius,
                    x + radius, y_pos + radius
                )
                self.root.after(20, lambda: animate(step + 1))
            else:
                self.canvas.delete(disc)
                self.update_board(board)
        
        animate()
    
    def run(self):
        """Start the GUI event loop (non-blocking)."""
        # Don't call mainloop - we'll update manually
        pass
    
    def mainloop(self):
        """Start blocking GUI event loop."""
        self.root.mainloop()
    
    def update(self):
        """Update the display (call this periodically)."""
        self.root.update()
    
    def destroy(self):
        """Close the window."""
        self.root.destroy()


# Example usage for testing
if __name__ == "__main__":
    def test_callback(col):
        print(f"Column {col} clicked!")
    
    display = BoardDisplay(rows=6, cols=7, on_column_click=test_callback)
    
    # Test: Update board with some pieces
    import numpy as np
    test_board = np.zeros((6, 7), dtype=int)
    test_board[5][0] = 1  # Human piece
    test_board[5][1] = 2  # Computer piece
    test_board[4][0] = 1  # Human piece
    
    display.update_board(test_board)
    display.run()

