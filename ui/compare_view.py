import customtkinter as ctk

from ui.theme import APP_BG, BTN_BG, BTN_HOVER, CARD_BG, MUTED, TEXT, TITLE


class CompareView(ctk.CTkFrame):
    def __init__(self, master, on_back):
        super().__init__(master, fg_color=APP_BG)
        self.on_back = on_back
        self.records = []

        self.build_ui()

    def build_ui(self):
        top = ctk.CTkFrame(self, fg_color=APP_BG)
        top.pack(fill="x", padx=20, pady=20)

        back_btn = ctk.CTkButton(
            top,
            text="← Configurar",
            command=self.on_back,
            fg_color=BTN_BG,
            hover_color=BTN_HOVER,
            width=120,
        )
        back_btn.pack(side="left")

        title = ctk.CTkLabel(
            self,
            text="Tabla Comparativa",
            text_color=TITLE,
            font=("Segoe UI", 28, "bold"),
        )
        title.pack(pady=(10, 20))

        self.textbox = ctk.CTkTextbox(
            self,
            fg_color=CARD_BG,
            text_color=TEXT,
            corner_radius=14,
            font=("Consolas", 14),
        )
        self.textbox.pack(fill="both", expand=True, padx=25, pady=20)
        self.textbox.insert("end", "Aún no hay resultados guardados.\n")

    def set_records(self, records):
        self.records = records
        self.render()

    def render(self):
        self.textbox.delete("1.0", "end")

        if not self.records:
            self.textbox.insert("end", "Aún no hay resultados guardados.\n")
            return

        header = f"{'Problema':<14} {'Algoritmo':<20} {'Heurística':<26} {'Tiempo(ms)':<12} {'Mem(KB)':<12} {'Nodos':<10} {'Pasos':<10} {'Resuelto':<10}\n"
        self.textbox.insert("end", header)
        self.textbox.insert("end", "-" * 120 + "\n")

        for r in self.records:
            line = f"{r['problema']:<14} {r['algoritmo']:<20} {r['heuristica']:<26} {r['tiempo_ms']:<12.2f} {r['memoria_kb']:<12.2f} {r['nodos']:<10} {r['pasos']:<10} {str(r['resuelto']):<10}\n"
            self.textbox.insert("end", line)