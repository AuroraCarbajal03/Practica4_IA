import customtkinter as ctk

from ui.compare_view import CompareView
from ui.home_view import HomeView
from ui.puzzle_view import PuzzleView
from ui.sudoku_view import SudokuView
from ui.theme import APP_BG


class SearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Búsqueda Informada - IA")
        self.geometry("1366x768")
        self.configure(fg_color=APP_BG)

        self.records = []
        self.current_view = None

        self.show_home()

    def clear_view(self):
        if self.current_view is not None:
            self.current_view.destroy()
            self.current_view = None

    def add_record(self, record):
        self.records.append(record)

    def show_home(self):
        self.clear_view()
        self.current_view = HomeView(
            self,
            on_open_puzzle=self.show_puzzle,
            on_open_sudoku=self.show_sudoku,
            on_open_compare=self.show_compare,
        )
        self.current_view.pack(fill="both", expand=True)

    def show_puzzle(self, size):
        self.clear_view()
        self.current_view = PuzzleView(
            self,
            size=size,
            on_back=self.show_home,
            add_record_callback=self.add_record
        )
        self.current_view.pack(fill="both", expand=True)

    def show_sudoku(self):
        self.clear_view()
        self.current_view = SudokuView(
            self,
            on_back=self.show_home,
            add_record_callback=self.add_record
        )
        self.current_view.pack(fill="both", expand=True)

    def show_compare(self):
        self.clear_view()
        self.current_view = CompareView(self, on_back=self.show_home)
        self.current_view.pack(fill="both", expand=True)
        self.current_view.set_records(self.records)