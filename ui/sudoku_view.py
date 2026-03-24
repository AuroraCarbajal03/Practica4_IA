import customtkinter as ctk

from algorithms.sudoku_astar import astar_sudoku
from algorithms.sudoku_sa import simulated_annealing_sudoku
from ui.theme import APP_BG, BTN_BG, BTN_HOVER, CARD_BG, DANGER, MUTED, SUCCESS, TEXT, TITLE
from utils.generators import generate_sudoku


class SudokuView(ctk.CTkFrame):
    def __init__(self, master, on_back, add_record_callback):
        super().__init__(master, fg_color=APP_BG)
        self.on_back = on_back
        self.add_record_callback = add_record_callback

        self.difficulty_var = ctk.StringVar(value="Fácil")
        self.initial_grid = generate_sudoku(20)
        self.current_grid = [row[:] for row in self.initial_grid]

        self.build_ui()
        self.draw_grid(self.initial_grid)

    def build_ui(self):
        top = ctk.CTkFrame(self, fg_color=APP_BG)
        top.pack(fill="x", padx=18, pady=18)

        ctk.CTkButton(
            top,
            text="← Configurar",
            command=self.on_back,
            fg_color=BTN_BG,
            hover_color=BTN_HOVER,
            width=120
        ).pack(side="left")

        ctk.CTkLabel(
            self,
            text="Sudoku",
            text_color=TITLE,
            font=("Segoe UI", 34, "bold")
        ).pack(pady=(0, 15))

        content = ctk.CTkFrame(self, fg_color=APP_BG)
        content.pack(fill="both", expand=True, padx=20, pady=10)

        left = ctk.CTkFrame(content, fg_color=APP_BG)
        left.pack(side="left", fill="both", expand=True)

        right = ctk.CTkFrame(content, fg_color=CARD_BG, corner_radius=16, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.canvas = ctk.CTkCanvas(left, width=540, height=540, bg=APP_BG, highlightthickness=0)
        self.canvas.pack(pady=10)

        controls = ctk.CTkFrame(left, fg_color=APP_BG)
        controls.pack(pady=12)

        ctk.CTkLabel(controls, text="Dificultad:", text_color=TEXT).grid(row=0, column=0, padx=6)
        ctk.CTkOptionMenu(
            controls,
            values=["Fácil", "Intermedio", "Difícil"],
            variable=self.difficulty_var,
            width=160
        ).grid(row=0, column=1, padx=6)

        ctk.CTkButton(controls, text="Nuevo Sudoku", command=self.new_sudoku, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=2, padx=6)
        ctk.CTkButton(controls, text="Resolver A*", command=self.solve_astar, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=3, padx=6)
        ctk.CTkButton(controls, text="Recocido Simulado", command=self.solve_sa, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=4, padx=6)
        ctk.CTkButton(controls, text="Comparar", command=self.compare_algorithms, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=5, padx=6)

        ctk.CTkLabel(
            right,
            text="Resultados",
            text_color=TITLE,
            font=("Segoe UI", 22, "bold")
        ).pack(anchor="w", padx=18, pady=(18, 12))

        self.results_box = ctk.CTkTextbox(
            right,
            fg_color=CARD_BG,
            text_color=TEXT,
            font=("Consolas", 14)
        )
        self.results_box.pack(fill="both", expand=True, padx=15, pady=10)
        self.results_box.insert("end", "Genera un sudoku y ejecuta un algoritmo.\n")

    def holes_by_difficulty(self):
        value = self.difficulty_var.get().lower()
        if value == "fácil" or value == "facil":
            return 20
        if value == "intermedio":
            return 35
        return 45

    def new_sudoku(self):
        holes = self.holes_by_difficulty()
        self.initial_grid = generate_sudoku(holes)
        self.current_grid = [row[:] for row in self.initial_grid]
        self.draw_grid(self.current_grid)
        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", f"Sudoku generado con {holes} celdas vacías.\n")

    def draw_grid(self, grid):
        self.canvas.delete("all")
        size = 540
        cell = size // 9
        offset = 10

        for r in range(9):
            for c in range(9):
                x1 = c * cell + offset
                y1 = r * cell + offset
                x2 = x1 + cell
                y2 = y1 + cell

                val = grid[r][c]
                fill = "#202545" if val == 0 else "#f0f4ff"
                text_color = MUTED if val == 0 else "#111827"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#6f7cb5", width=1)
                if val != 0:
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=str(val), fill=text_color, font=("Segoe UI", 18, "bold"))

        for i in range(10):
            w = 3 if i % 3 == 0 else 1
            self.canvas.create_line(offset, offset + i * cell, offset + 9 * cell, offset + i * cell, fill="#93a4ea", width=w)
            self.canvas.create_line(offset + i * cell, offset, offset + i * cell, offset + 9 * cell, fill="#93a4ea", width=w)

    def show_result(self, title, result, algorithm_name):
        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", f"{title}\n\n")
        self.results_box.insert("end", f"Algoritmo: {algorithm_name}\n")
        self.results_box.insert("end", f"Tiempo: {result['time_ms']:.2f} ms\n")
        self.results_box.insert("end", f"Memoria: {result['memory_kb']:.2f} KB\n")
        self.results_box.insert("end", f"Nodos / iteraciones: {result['nodes']}\n")
        self.results_box.insert("end", f"Pasos: {result['steps']}\n")
        self.results_box.insert("end", f"Resuelto: {'Sí' if result['solved'] else 'No'}\n")

    def solve_astar(self):
        result = astar_sudoku(self.initial_grid)
        if result["solution"] is not None:
            self.current_grid = result["solution"]
            self.draw_grid(self.current_grid)

        self.show_result("Sudoku - A*", result, "A*")

        self.add_record_callback({
            "problema": f"sudoku-{self.difficulty_var.get().lower()}",
            "algoritmo": "A*",
            "heuristica": "MRV + vacíos restantes",
            "tiempo_ms": result["time_ms"],
            "memoria_kb": result["memory_kb"],
            "nodos": result["nodes"],
            "pasos": result["steps"],
            "resuelto": result["solved"],
        })

    def solve_sa(self):
        result = simulated_annealing_sudoku(self.initial_grid)
        if result["solution"] is not None:
            self.current_grid = result["solution"]
            self.draw_grid(self.current_grid)

        self.show_result("Sudoku - Recocido Simulado", result, "Recocido Simulado")

        self.add_record_callback({
            "problema": f"sudoku-{self.difficulty_var.get().lower()}",
            "algoritmo": "Recocido Simulado",
            "heuristica": "Vecindad por subcuadros",
            "tiempo_ms": result["time_ms"],
            "memoria_kb": result["memory_kb"],
            "nodos": result["nodes"],
            "pasos": result["steps"],
            "resuelto": result["solved"],
        })

    def compare_algorithms(self):
        result_a = astar_sudoku(self.initial_grid)
        result_b = simulated_annealing_sudoku(self.initial_grid)

        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", "Comparación de algoritmos\n\n")

        self.results_box.insert("end", "A*\n")
        self.results_box.insert("end", f"  Tiempo: {result_a['time_ms']:.2f} ms\n")
        self.results_box.insert("end", f"  Memoria: {result_a['memory_kb']:.2f} KB\n")
        self.results_box.insert("end", f"  Nodos: {result_a['nodes']}\n")
        self.results_box.insert("end", f"  Pasos: {result_a['steps']}\n")
        self.results_box.insert("end", f"  Resuelto: {'Sí' if result_a['solved'] else 'No'}\n\n")

        self.results_box.insert("end", "Recocido Simulado\n")
        self.results_box.insert("end", f"  Tiempo: {result_b['time_ms']:.2f} ms\n")
        self.results_box.insert("end", f"  Memoria: {result_b['memory_kb']:.2f} KB\n")
        self.results_box.insert("end", f"  Nodos: {result_b['nodes']}\n")
        self.results_box.insert("end", f"  Pasos: {result_b['steps']}\n")
        self.results_box.insert("end", f"  Resuelto: {'Sí' if result_b['solved'] else 'No'}\n\n")

        self.add_record_callback({
            "problema": f"sudoku-{self.difficulty_var.get().lower()}",
            "algoritmo": "A*",
            "heuristica": "MRV + vacíos restantes",
            "tiempo_ms": result_a["time_ms"],
            "memoria_kb": result_a["memory_kb"],
            "nodos": result_a["nodes"],
            "pasos": result_a["steps"],
            "resuelto": result_a["solved"],
        })

        self.add_record_callback({
            "problema": f"sudoku-{self.difficulty_var.get().lower()}",
            "algoritmo": "Recocido Simulado",
            "heuristica": "Vecindad por subcuadros",
            "tiempo_ms": result_b["time_ms"],
            "memoria_kb": result_b["memory_kb"],
            "nodos": result_b["nodes"],
            "pasos": result_b["steps"],
            "resuelto": result_b["solved"],
        })