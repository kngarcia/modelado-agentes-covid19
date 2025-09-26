# agentes_ui_controls_modificado.py
import enum
import numpy as np
import pandas as pd
from io import BytesIO

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Category10

import panel as pn
pn.extension()

# ---------- Modelo ----------
class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

class InfectionModel(Model):
    def __init__(self, N=200, width=20, height=20, ptrans=0.25,
                 death_rate=0.01, recovery_days=21, recovery_sd=7,
                 vaccination_rate=0.0, mask_effect=1.0, isolation_prob=0.0,
                 infection_radius=1, seed=None):
        """
        vaccination_rate: porcentaje inicial de vacunados (inmunes).
        mask_effect: factor de reducción de ptrans (ej. 0.5 = 50% menos).
        isolation_prob: probabilidad de que un agente NO se mueva.
        infection_radius: radio de distancia de contagio (1 = vecinos Moore).
        """
        super().__init__(seed=seed)
        self.num_agents = N
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.ptrans = ptrans
        self.death_rate = death_rate
        self.vaccination_rate = vaccination_rate
        self.mask_effect = mask_effect
        self.isolation_prob = isolation_prob
        self.infection_radius = infection_radius
        self.grid = MultiGrid(width, height, torus=True)
        self.dead_agents = []

        for _ in range(self.num_agents):
            a = MyAgent(self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

            # Inicializar vacunados/inmunes
            if self.random.random() < self.vaccination_rate:
                a.state = State.REMOVED
            # Semilla de infección inicial
            elif self.random.random() < 0.02:
                a.state = State.INFECTED
                a.recovery_time = self.get_recovery_time()
                a.infection_time = self.steps

        self.datacollector = DataCollector(agent_reporters={"State": lambda a: int(a.state)})

    def get_recovery_time(self):
        val = int(self.random.normalvariate(self.recovery_days, self.recovery_sd))
        return max(1, val)

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

class MyAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.age = max(0, self.random.normalvariate(20, 40))
        self.state = State.SUSCEPTIBLE
        self.infection_time = None
        self.recovery_time = None

    def move(self):
        # Probabilidad de quedarse quieto (aislamiento)
        if self.random.random() < self.model.isolation_prob:
            return
        neigh = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=1
        )
        if neigh:
            new_pos = self.random.choice(neigh)
            self.model.grid.move_agent(self, new_pos)

    def status(self):
        if self.state == State.INFECTED:
            if self.random.random() < self.model.death_rate:
                try:
                    self.remove()
                except Exception:
                    try:
                        self.model.grid.remove_agent(self)
                    except Exception:
                        pass
                try:
                    self.model.dead_agents.append(getattr(self, "unique_id", None))
                except Exception:
                    pass
                return
            if self.infection_time is not None:
                t = self.model.steps - self.infection_time
                if t >= (self.recovery_time or 0):
                    self.state = State.REMOVED

    def contact(self):
        # Revisar vecinos dentro del radio de contagio
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True,
                                                  include_center=False,
                                                  radius=self.model.infection_radius)
        for other in neighbors:
            if self.state == State.INFECTED and other.state == State.SUSCEPTIBLE:
                # Aplicar efecto de tapabocas (reduce ptrans)
                effective_ptrans = self.model.ptrans * self.model.mask_effect
                if self.random.random() < effective_ptrans:
                    other.state = State.INFECTED
                    other.infection_time = self.model.steps
                    other.recovery_time = self.model.get_recovery_time()

    def step(self):
        self.status()
        if self.state == State.INFECTED:
            self.contact()
        self.move()

# ---------- Utilidades ----------
def count_states(model):
    s = i = r = 0
    for a in model.agents:
        if a.state == State.SUSCEPTIBLE:
            s += 1
        elif a.state == State.INFECTED:
            i += 1
        elif a.state == State.REMOVED:
            r += 1
    return s, i, r

def grid_values_stack(model):
    w = model.grid.width
    h = model.grid.height
    arr = np.full((w, h), -1, dtype=int)
    for cell in model.grid.coord_iter():
        if len(cell) == 3:
            agents, x, y = cell
        else:
            agents, pos = cell
            x, y = pos
        val = -1
        for a in agents:
            val = int(a.state)
        if 0 <= x < w and 0 <= y < h:
            arr[x, y] = val
    df = pd.DataFrame(arr)
    stacked = df.stack().reset_index()
    stacked.columns = ['x', 'y', 'value']
    return stacked

def get_column_data(model):
    df = model.datacollector.get_agent_vars_dataframe()
    if df.empty:
        return pd.DataFrame(columns=['Susceptible','Infected','Removed'])
    df = df.reset_index()
    if 'State' not in df.columns:
        possible = [c for c in df.columns if isinstance(c, str) and 'State' in c]
        if possible:
            df['State'] = df[possible[0]]
        else:
            return pd.DataFrame(columns=['Susceptible','Infected','Removed'])
    df['State'] = df['State'].astype(int)
    grouped = df.groupby(['Step','State']).size().unstack(fill_value=0)
    mapping = {0:'Susceptible',1:'Infected',2:'Removed'}
    new_cols = {}
    for col in grouped.columns:
        try:
            c_int = int(col)
            new_cols[col] = mapping.get(c_int,str(c_int))
        except Exception:
            new_cols[col] = str(col)
    grouped = grouped.rename(columns=new_cols)
    for col in ['Susceptible','Infected','Removed']:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped = grouped[['Susceptible','Infected','Removed']]
    grouped.index.name = 'Step'
    grouped = grouped.sort_index()
    return grouped

# ---------- Bokeh plots ----------
def create_time_series_plot():
    source = ColumnDataSource(data=dict(Step=[], Susceptible=[], Infected=[], Removed=[]))
    p = figure(width=700, height=420, title="Estados (tiempo)", tools="pan,wheel_zoom,box_zoom,reset")
    palette = list(Category10[3])
    p.line(x='Step', y='Susceptible', source=source, line_width=3, legend_label='Susceptible', color=palette[0])
    p.line(x='Step', y='Infected', source=source, line_width=3, legend_label='Infected', color=palette[1])
    p.line(x='Step', y='Removed', source=source, line_width=3, legend_label='Removed', color=palette[2])
    p.xaxis.axis_label = 'Step'
    p.yaxis.axis_label = 'Count'
    p.legend.location = "top_right"
    p.toolbar.logo = None
    return p, source

def create_grid_plot(model):
    stacked = grid_values_stack(model)
    source = ColumnDataSource(stacked)
    base_palette = list(Category10[3])
    full_palette = ["#d9d9d9"] + base_palette
    mapper = LinearColorMapper(palette=full_palette, low=-1, high=2)
    p = figure(width=500, height=500, title="Grid espacial", tools="hover")
    hover = p.select_one(HoverTool)
    hover.tooltips = [("x","@x"), ("y","@y"), ("state","@value")]
    p.rect(x='x', y='y', width=1, height=1, source=source,
           fill_color={'field':'value','transform':mapper}, line_color='black')
    p.x_range.start = -0.5
    p.x_range.end = model.grid.width - 0.5
    p.y_range.start = -0.5
    p.y_range.end = model.grid.height - 0.5
    p.axis.visible = False
    p.grid.grid_line_color = None
    p.toolbar.logo = None
    return p, source

# ---------- App ----------
def build_app(pop=300, width=20, height=20, interval_ms=200, print_every=10):
    # Parámetros iniciales
    model = InfectionModel(N=pop, width=width, height=height,
                           ptrans=0.25, death_rate=0.01,
                           vaccination_rate=0.2, mask_effect=0.5,
                           isolation_prob=0.3, infection_radius=1, seed=42)

    ts_plot, ts_source = create_time_series_plot()
    grid_plot, grid_source = create_grid_plot(model)

    # initial counts
    s,i,r = count_states(model)
    ts_source.data = dict(Step=[0], Susceptible=[s], Infected=[i], Removed=[r])

    # widgets
    pause = pn.widgets.Toggle(name='Pausar / Reanudar', value=False, button_type='primary')
    step_btn = pn.widgets.Button(name='Step →', button_type='success')
    reset_btn = pn.widgets.Button(name='Reset', button_type='danger')
    download_btn = pn.widgets.FileDownload(label='Descargar CSV (tabla completa)', filename='sim_results.csv')
    interval_slider = pn.widgets.IntSlider(name='Interval (ms)', start=50, end=2000, step=10, value=interval_ms)
    print_every_input = pn.widgets.IntInput(name='Print every N steps (0=off)', value=print_every)

    # nuevos parámetros ajustables
    vacc_slider = pn.widgets.FloatSlider(name='Tasa de vacunación', start=0, end=1, step=0.05, value=0.2)
    mask_slider = pn.widgets.FloatSlider(name='Efecto tapabocas (0-1)', start=0, end=1, step=0.05, value=0.5)
    iso_slider = pn.widgets.FloatSlider(name='Prob. de aislamiento', start=0, end=1, step=0.05, value=0.3)
    radius_slider = pn.widgets.IntSlider(name='Radio de infección', start=1, end=5, step=1, value=1)
    ptrans_slider = pn.widgets.FloatSlider(name='Prob. transmisión base', start=0, end=1, step=0.05, value=0.25)

    # status pane
    status = pn.pane.Markdown("**Estado:** listo", width=300)

    # update function
    def update():
        if pause.value:
            return
        model.step()
        s,i,r = count_states(model)
        ts_source.stream(dict(Step=[model.steps], Susceptible=[s], Infected=[i], Removed=[r]), rollover=100000)
        stacked = grid_values_stack(model)
        grid_source.data = ColumnDataSource.from_df(stacked)
        status.object = f"**Estado:** step={model.steps}  |  S={s} I={i} R={r}"

    callback = pn.state.add_periodic_callback(update, interval_slider.value)

    def on_interval_change(event):
        pn.state.periodic_callbacks.clear()
        pn.state.add_periodic_callback(update, interval_slider.value)
    interval_slider.param.watch(on_interval_change, 'value')

    def on_step(event):
        model.step()
        s,i,r = count_states(model)
        ts_source.stream(dict(Step=[model.steps], Susceptible=[s], Infected=[i], Removed=[r]), rollover=100000)
        stacked = grid_values_stack(model)
        grid_source.data = ColumnDataSource.from_df(stacked)
        status.object = f"**Estado:** step={model.steps}  |  S={s} I={i} R={r}"
    step_btn.on_click(on_step)

    def on_reset(event):
        nonlocal model
        model = InfectionModel(N=pop, width=width, height=height,
                               ptrans=ptrans_slider.value,
                               vaccination_rate=vacc_slider.value,
                               mask_effect=mask_slider.value,
                               isolation_prob=iso_slider.value,
                               infection_radius=radius_slider.value,
                               death_rate=0.01, seed=None)
        s,i,r = count_states(model)
        ts_source.data = dict(Step=[0], Susceptible=[s], Infected=[i], Removed=[r])
        stacked = grid_values_stack(model)
        grid_source.data = ColumnDataSource.from_df(stacked)
        status.object = f"**Estado:** reiniciado. step=0  |  S={s} I={i} R={r}"
    reset_btn.on_click(on_reset)

    def get_csv():
        pivot = get_column_data(model)
        b = pivot.to_csv(index=True).encode('utf-8')
        return b
    download_btn.callback = get_csv

    # controles
    left_col = pn.Column(pn.pane.Markdown("### Controles"),
                         pn.Row(pause, step_btn, reset_btn),
                         pn.Row(download_btn, print_every_input))
    right_col = pn.Column(pn.pane.Markdown("### Parámetros"),
                          interval_slider,
                          ptrans_slider, vacc_slider, mask_slider,
                          iso_slider, radius_slider,
                          pn.Row(pn.pane.Markdown("Población:"), pn.pane.Str(f"{pop}")))
    controls = pn.Row(left_col, right_col, status, sizing_mode='stretch_width')

    layout = pn.Column("# Simulación interactiva", controls, pn.Row(ts_plot, grid_plot))
    return layout

app = build_app(pop=400, width=20, height=20, interval_ms=200, print_every=10)
app.servable()
