# Practica 2 - Inteligencia Artificial

Aplicacion de escritorio para resolver puzzles mediante algoritmos de busqueda de IA. Implementa el puzzle de 8 piezas y Sudoku utilizando A* y Recocido Simulado (Simulated Annealing).

## Algoritmos implementados

- **A\*** (A-Star) — para el puzzle de 8 piezas y Sudoku, con heuristicas configurables
- **Recocido Simulado** (Simulated Annealing) — para resolver Sudoku
- **Heuristicas** — Distancia Manhattan y otras funciones de estimacion

## Tecnologias

- Python 3.12
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — interfaz grafica moderna
- [psutil](https://github.com/giampaolo/psutil) — metricas de rendimiento del sistema

## Estructura del proyecto

```
Practica 2 IA/
├── main.py              # Punto de entrada
├── requirements.txt     # Dependencias
├── algorithms/          # Implementaciones de algoritmos de busqueda
│   ├── astar_puzzle.py
│   ├── sudoku_astar.py
│   ├── sudoku_sa.py
│   └── heuristics.py
├── models/              # Representacion de estados
│   ├── puzzle_state.py
│   └── sudoku_state.py
├── ui/                  # Interfaz grafica
│   ├── app.py
│   ├── home_view.py
│   ├── puzzle_view.py
│   ├── sudoku_view.py
│   ├── compare_view.py
│   └── theme.py
└── utils/               # Utilidades
    ├── generators.py
    ├── metrics.py
    └── validators.py
```

## Instalacion

1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd "Practica 2 IA"
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

## Uso

```bash
python main.py
```
