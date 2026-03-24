import customtkinter as ctk

from ui.theme import APP_BG, BTN_BG, BTN_HOVER, CARD_BG, TEXT, TITLE


class HomeView(ctk.CTkFrame):
    def __init__(self, master, on_open_puzzle, on_open_sudoku, on_open_compare):
        super().__init__(master, fg_color=APP_BG)
        self.on_open_puzzle = on_open_puzzle
        self.on_open_sudoku = on_open_sudoku
        self.on_open_compare = on_open_compare

        self.build_ui()

    def build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="Búsqueda Informada - IA",
            text_color=TITLE,
            font=("Segoe UI", 32, "bold"),
        )
        title.pack(pady=(40, 10))

        subtitle = ctk.CTkLabel(
            self,
            text="A* para 8-Puzzle y 15-Puzzle | A* y Recocido Simulado para Sudoku",
            text_color=TEXT,
            font=("Segoe UI", 16),
        )
        subtitle.pack(pady=(0, 30))

        container = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=18)
        container.pack(padx=40, pady=20, fill="both", expand=True)

        buttons = [
            ("8-Puzzle", lambda: self.on_open_puzzle(3)),
            ("15-Puzzle", lambda: self.on_open_puzzle(4)),
            ("Sudoku", self.on_open_sudoku),
            ("Comparativas", self.on_open_compare),
        ]

        for text, cmd in buttons:
            btn = ctk.CTkButton(
                container,
                text=text,
                command=cmd,
                height=52,
                width=260,
                fg_color=BTN_BG,
                hover_color=BTN_HOVER,
                font=("Segoe UI", 18, "bold"),
                corner_radius=12,
            )
            btn.pack(pady=18)