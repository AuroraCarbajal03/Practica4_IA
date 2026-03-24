import customtkinter as ctk

from algorithms.astar_puzzle import astar_puzzle
from algorithms.heuristics import get_heuristic
from ui.theme import APP_BG, BTN_BG, BTN_HOVER, CARD_BG, MUTED, TEXT, TILE_COLORS, TITLE
from utils.generators import shuffle_puzzle


class PuzzleView(ctk.CTkFrame):
    def __init__(self, master, size, on_back, add_record_callback):
        super().__init__(master, fg_color=APP_BG)
        self.size = size
        self.on_back = on_back
        self.add_record_callback = add_record_callback

        self.goal = tuple(list(range(1, size * size)) + [0])
        self.start = shuffle_puzzle(self.goal, size, steps=40 if size == 3 else 60)
        self.current_path = []
        self.current_moves = []
        self.current_index = 0

        self.heuristic_var = ctk.StringVar(value="Distancia Manhattan")
        self.speed_var = ctk.IntVar(value=2)

        self.build_ui()
        self.draw_boards()

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

        ctk.CTkButton(
            top,
            text="Comparar heurísticas",
            command=self.compare_heuristics,
            fg_color=BTN_BG,
            hover_color=BTN_HOVER,
            width=180
        ).pack(side="right")

        title_text = "8-Puzzle" if self.size == 3 else "15-Puzzle"
        ctk.CTkLabel(
            self,
            text=title_text,
            text_color=TITLE,
            font=("Segoe UI", 34, "bold")
        ).pack(pady=(0, 15))

        content = ctk.CTkFrame(self, fg_color=APP_BG)
        content.pack(fill="both", expand=True, padx=20, pady=10)

        left = ctk.CTkFrame(content, fg_color=APP_BG)
        left.pack(side="left", fill="both", expand=True)

        right = ctk.CTkFrame(content, fg_color=CARD_BG, corner_radius=16, width=310)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        boards_frame = ctk.CTkFrame(left, fg_color=APP_BG)
        boards_frame.pack(pady=10)

        self.initial_canvas = ctk.CTkCanvas(boards_frame, width=220, height=220, bg=APP_BG, highlightthickness=0)
        self.initial_canvas.grid(row=0, column=0, padx=10)

        self.current_canvas = ctk.CTkCanvas(boards_frame, width=220, height=220, bg=APP_BG, highlightthickness=0)
        self.current_canvas.grid(row=0, column=1, padx=10)

        self.goal_canvas = ctk.CTkCanvas(boards_frame, width=220, height=220, bg=APP_BG, highlightthickness=0)
        self.goal_canvas.grid(row=0, column=2, padx=10)

        labels_frame = ctk.CTkFrame(left, fg_color=APP_BG)
        labels_frame.pack()

        ctk.CTkLabel(labels_frame, text="Inicial", text_color=MUTED, font=("Segoe UI", 16)).grid(row=0, column=0, padx=70)
        ctk.CTkLabel(labels_frame, text="Actual", text_color=MUTED, font=("Segoe UI", 16)).grid(row=0, column=1, padx=70)
        ctk.CTkLabel(labels_frame, text="Objetivo", text_color=MUTED, font=("Segoe UI", 16)).grid(row=0, column=2, padx=70)

        controls = ctk.CTkFrame(left, fg_color=APP_BG)
        controls.pack(pady=20)

        ctk.CTkButton(controls, text="Resolver A*", command=self.solve_current, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=0, padx=8)
        ctk.CTkButton(controls, text="Paso anterior", command=self.prev_step, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=1, padx=8)
        ctk.CTkButton(controls, text="Paso siguiente", command=self.next_step, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=2, padx=8)
        ctk.CTkButton(controls, text="Nuevo tablero", command=self.new_board, fg_color=BTN_BG, hover_color=BTN_HOVER).grid(row=0, column=3, padx=8)

        bottom = ctk.CTkFrame(left, fg_color=APP_BG)
        bottom.pack(pady=8)

        ctk.CTkLabel(bottom, text="Heurística:", text_color=TEXT).pack(side="left", padx=5)
        ctk.CTkOptionMenu(
            bottom,
            values=["Fichas fuera de lugar", "Distancia Manhattan", "Heurística personalizada"],
            variable=self.heuristic_var,
            width=220
        ).pack(side="left", padx=8)

        ctk.CTkLabel(bottom, text="Velocidad:", text_color=TEXT).pack(side="left", padx=(20, 5))
        ctk.CTkSlider(bottom, from_=1, to=10, number_of_steps=9, variable=self.speed_var, width=160).pack(side="left")

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
        self.results_box.insert("end", "Selecciona una heurística y presiona Resolver A*.\n")

    def draw_board(self, canvas, board):
        canvas.delete("all")
        cell = 220 // self.size
        pad = 4

        for i, value in enumerate(board):
            r, c = divmod(i, self.size)
            x1 = c * cell + pad
            y1 = r * cell + pad
            x2 = x1 + cell - 2 * pad
            y2 = y1 + cell - 2 * pad

            if value == 0:
                color = "#202545"
                text = ""
            else:
                color = TILE_COLORS[(value - 1) % len(TILE_COLORS)]
                text = str(value)

            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#3c4574", width=2)
            canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill="#111827", font=("Segoe UI", 20, "bold"))

    def draw_boards(self):
        self.draw_board(self.initial_canvas, self.start)
        current = self.start if not self.current_path else self.current_path[self.current_index]
        self.draw_board(self.current_canvas, current)
        self.draw_board(self.goal_canvas, self.goal)

    def show_result(self, title, result, heuristic_name):
        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", f"{title}\n\n")
        self.results_box.insert("end", f"Heurística: {heuristic_name}\n")
        self.results_box.insert("end", f"Tiempo: {result['time_ms']:.2f} ms\n")
        self.results_box.insert("end", f"Memoria: {result['memory_kb']:.2f} KB\n")
        self.results_box.insert("end", f"Nodos: {result['nodes']}\n")
        self.results_box.insert("end", f"Pasos: {result['steps']}\n")
        self.results_box.insert("end", f"Resuelto: {'Sí' if result['solved'] else 'No'}\n\n")

        if result["moves"]:
            self.results_box.insert("end", "Movimientos:\n")
            for i, move in enumerate(result["moves"], start=1):
                self.results_box.insert("end", f"{i}. {move}\n")

    def solve_current(self):
        heuristic_name = self.heuristic_var.get()
        heuristic_fn = get_heuristic(heuristic_name)

        result = astar_puzzle(self.start, self.goal, self.size, heuristic_fn)

        self.current_path = result["path"]
        self.current_moves = result["moves"]
        self.current_index = 0
        self.draw_boards()
        self.show_result("A* - Resultado", result, heuristic_name)

        self.add_record_callback({
            "problema": "8-puzzle" if self.size == 3 else "15-puzzle",
            "algoritmo": "A*",
            "heuristica": heuristic_name,
            "tiempo_ms": result["time_ms"],
            "memoria_kb": result["memory_kb"],
            "nodos": result["nodes"],
            "pasos": result["steps"],
            "resuelto": result["solved"],
        })

    def compare_heuristics(self):
        heuristics = [
            "Fichas fuera de lugar",
            "Distancia Manhattan",
            "Heurística personalizada",
        ]

        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", "Comparación de heurísticas\n\n")

        for h_name in heuristics:
            fn = get_heuristic(h_name)
            result = astar_puzzle(self.start, self.goal, self.size, fn)

            self.results_box.insert("end", f"{h_name}\n")
            self.results_box.insert("end", f"  Tiempo: {result['time_ms']:.2f} ms\n")
            self.results_box.insert("end", f"  Memoria: {result['memory_kb']:.2f} KB\n")
            self.results_box.insert("end", f"  Nodos: {result['nodes']}\n")
            self.results_box.insert("end", f"  Pasos: {result['steps']}\n")
            self.results_box.insert("end", f"  Resuelto: {'Sí' if result['solved'] else 'No'}\n\n")

            self.add_record_callback({
                "problema": "8-puzzle" if self.size == 3 else "15-puzzle",
                "algoritmo": "A*",
                "heuristica": h_name,
                "tiempo_ms": result["time_ms"],
                "memoria_kb": result["memory_kb"],
                "nodos": result["nodes"],
                "pasos": result["steps"],
                "resuelto": result["solved"],
            })

    def prev_step(self):
        if self.current_path and self.current_index > 0:
            self.current_index -= 1
            self.draw_boards()

    def next_step(self):
        if self.current_path and self.current_index < len(self.current_path) - 1:
            self.current_index += 1
            self.draw_boards()

    def new_board(self):
        self.start = shuffle_puzzle(self.goal, self.size, steps=40 if self.size == 3 else 60)
        self.current_path = []
        self.current_moves = []
        self.current_index = 0
        self.draw_boards()
        self.results_box.delete("1.0", "end")
        self.results_box.insert("end", "Nuevo tablero generado.\n")