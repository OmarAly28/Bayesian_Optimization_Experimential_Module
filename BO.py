import numpy as np
import threading
import warnings
from functools import partial
import csv
import io
import base64

from skopt import Optimizer
from skopt.space import Real
from skopt.learning import (
    GaussianProcessRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Select,
    Div,
    Spinner,
    TextInput,
    RadioButtonGroup,
    DataTable,
    TableColumn,
    NumberFormatter,
    ColumnDataSource,
    FileInput,
)

# --- Suppress scikit-optimize warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global State ---
optimizer = None
param_names = []
dimensions = []
maximize_objective = True
experiment_history = []
suggested_x = None
TOLERANCE = 1e-4    # Stopping threshold
initial_loaded_data = [] # Global list to store parsed CSV data

# --- Bokeh Application Setup ---
doc = curdoc()
doc.title = "Interactive Optimizer"

# --- Data Sources ---
experiments_source = ColumnDataSource(data=dict(Iteration=[], Objective=[]))
convergence_source = ColumnDataSource(data=dict(iter=[], best_value=[]))

# --- UI and Workflow Functions ---
def set_ui_state(phase='setup', lock_all=False):
    """Manages which UI elements are enabled or disabled."""
    if lock_all:
        for widget in all_buttons + setup_widgets + optimization_widgets:
            widget.disabled = True
        return

    is_setup = (phase == 'setup')
    for widget in setup_widgets:
        widget.disabled = not is_setup
    for widget in optimization_widgets:
        widget.disabled = is_setup

    reset_button.disabled = is_setup
    submit_result_button.disabled = True
    # Enable/disable CSV upload based on setup phase
    csv_file_input.disabled = not is_setup

def update_direction_indicator():
    """Updates the direction indicator text based on current setting"""
    direction = "Maximizing" if maximize_objective else "Minimizing"
    direction_indicator.text = f"<i>({direction})</i>"
    direction_indicator.styles = {
        'color': 'green' if maximize_objective else 'blue',
        'font-weight': 'bold'
    }

def on_num_params_change(attr, old, new):
    """Shows/hides parameter definition rows and updates initial data rows."""
    for i in range(MAX_PARAMS):
        param_rows[i].visible = (i < new)
        initial_data_headers[i].visible = (i < new)
        if i < new:
            on_param_range_change(i, None, None, None)
    # Ensure initial data rows are updated based on spinner value
    on_initial_data_change(None, None, initial_data_spinner.value) 
    update_initial_random_points_info()

def on_initial_data_change(attr, old, new):
    """Shows/hides rows for entering initial data."""
    num_to_show = new or 0
    initial_data_header_row.visible = (num_to_show > 0)
    active_param_indices = [i for i in range(num_params_spinner.value)]
    for i in range(MAX_INITIAL_POINTS):
        row_visible = i < num_to_show
        initial_data_rows[i].visible = row_visible
        if row_visible:
            for j in range(MAX_PARAMS):
                initial_param_inputs[i][j].visible = (j in active_param_indices)
            
    update_initial_points_warning(new)
    update_initial_random_points_info()

def on_objective_name_change(attr, old, new):
    """Updates the initial data header for the objective column in real-time."""
    objective_header.text = f"<b>{new or 'Objective'}</b>"
    update_direction_indicator()

def on_param_name_change(index, attr, old, new):
    """Updates a specific parameter header in the initial data section in real-time."""
    initial_data_headers[index].text = f"<b>{new or f'Param {index + 1}'}</b>"

def on_surrogate_model_change(attr, old, new):
    """Updates acquisition function options based on the selected model."""
    if new == "GP":
        acq_func_select.options = ["gp_hedge", "EI", "PI", "LCB"]
    else:
        acq_func_select.options = ["EI", "PI", "LCB"]
        if acq_func_select.value == "gp_hedge":
            acq_func_select.value = "EI"

def on_param_range_change(index, attr, old, new):
    """Updates the min/max of the initial data spinners when a parameter's range is changed."""
    low_val = param_low_spinners[index].value
    high_val = param_high_spinners[index].value

    if low_val is None or high_val is None:
        return

    for k in range(MAX_INITIAL_POINTS):
        spinner_to_update = initial_param_inputs[k][index]
        spinner_to_update.low = low_val
        spinner_to_update.high = high_val
        if spinner_to_update.value is not None:
            spinner_to_update.value = max(low_val, min(high_val, spinner_to_update.value))

def update_initial_points_warning(num_points=0):
    """Shows a warning if the number of points is between 1 and 4."""
    num_points = num_points or 0
    initial_points_warning_div.visible = 0 < num_points < 5

def update_initial_random_points_info():
    """Updates the info div about the number of initial random points."""
    num_params = num_params_spinner.value
    num_initial_data = initial_data_spinner.value
    
    # Calculate the effective n_initial_points that skopt will use
    effective_initial_random_points = max(5, 2 * num_params, num_initial_data)

    
    info_text = f"""
    <p style="font-size: 0.9em; color: #555;">
    The optimizer requires at least <b>{effective_initial_random_points}</b> total initial experiments
    to build its first reliable model before suggesting optimized points.
    <br><br>
    This count includes any existing data you provide. If you provide fewer than this number,
    the optimizer will automatically suggest additional random experiments
    until this total is reached. This initial phase helps explore the search space effectively.
    </p>
    """
    initial_random_points_info_div.text = info_text


def lock_in_setup():
    """Reads all setup widgets, creates the Optimizer, and transitions the UI."""
    global initial_loaded_data # Access the global variable
    update_status("🔄 Initializing...")
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    try:
        config = {
            "num_params": num_params_spinner.value,
            "objective_name": objective_name_input.value or "Objective",
            "maximize": objective_type_select.active == 0,
            "surrogate_model": surrogate_select.value,
            "acq_func": acq_func_select.value,
            "num_initial_data": initial_data_spinner.value, # This value is set by CSV upload or manual input
            "params": [],
            "initial_data": [] # This will be populated from initial_loaded_data
        }

        # IMPROVEMENT: Make n_initial_points more robust based on num_params
        # This ensures better initial exploration for higher-dimensional problems.
        config["initial_random_points"] = max(5, 2 * config["num_params"], config["num_initial_data"])

        p_names_check = set()
        for i in range(config["num_params"]):
            name = param_name_inputs[i].value.strip()
            if not name or name in p_names_check:
                raise ValueError(f"Invalid or duplicate name for Parameter {i+1}")
            p_names_check.add(name)

            low = param_low_spinners[i].value
            high = param_high_spinners[i].value
            if low is None or high is None or low >= high:
                raise ValueError(f"Invalid range for '{name}'.")

            config["params"].append({"name": name, "low": low, "high": high})

        if not config["params"]:
            raise ValueError("At least one parameter must be defined.")

        # Use the globally stored initial_loaded_data directly for the optimizer
        # This bypasses potential timing issues with reading from UI spinners
        config["initial_data"] = list(initial_loaded_data) # Create a copy

        print(f"DEBUG: In lock_in_setup - config['initial_data'] before optimizer.tell: {config['initial_data']}")


    except Exception as e:
        update_status(f"❌ Error: {e}", is_error=True)
        set_ui_state(phase='setup')
        return

    def worker(config):
        global optimizer, dimensions, param_names, maximize_objective, experiment_history
        try:
            doc.add_next_tick_callback(lambda: update_status("🔄 Step 1/4: Validating parameters..."))

            dims = [Real(p['low'], p['high'], name=p['name']) for p in config['params']]
            p_names = [p['name'] for p in config['params']]

            table_cols = [
                TableColumn(field="Iteration", title="Iteration"),
                TableColumn(field="Objective", title=config['objective_name'], formatter=NumberFormatter(format="0.0000")),
            ]
            for name in p_names:
                table_cols.append(TableColumn(field=name, title=name, formatter=NumberFormatter(format="0.0000")))

            doc.add_next_tick_callback(lambda: update_status("🔄 Step 2/4: Initializing optimizer model..."))

            param_names, dimensions, maximize_objective = p_names, dims, config['maximize']
            
            # Update direction indicator
            doc.add_next_tick_callback(update_direction_indicator)

            def update_table_cols():
                data_table.columns = table_cols
            doc.add_next_tick_callback(update_table_cols)

            model_map = {"GP": GaussianProcessRegressor(normalize_y=True), "RF": RandomForestRegressor(n_estimators=100), "ET": ExtraTreesRegressor(n_estimators=100)}

            optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator=model_map[config['surrogate_model']],
                acq_func=config['acq_func'],
                n_initial_points=config['initial_random_points']
            )

            doc.add_next_tick_callback(lambda: update_status("🔄 Step 3/4: Processing initial data..."))

            experiment_history = []
            initial_xs, initial_ys = [], []
            if config["initial_data"]:
                for point in config["initial_data"]:
                    x, y = point["x"], point["y"]
                    initial_xs.append(x)
                    internal_y = -y if maximize_objective else y
                    initial_ys.append(internal_y)
                    experiment_history.append((x, internal_y))

                optimizer.tell(initial_xs, initial_ys)
                print(f"DEBUG: Optimizer told {len(initial_xs)} initial points.")
            else:
                print("DEBUG: No initial data provided to optimizer.")


            doc.add_next_tick_callback(lambda: update_status("🔄 Step 4/4: Finalizing setup..."))

            def final_callback():
                new_cols = {name: [] for name in param_names}
                experiments_source.data.update(new_cols)

                if initial_xs:
                    new_data = {"Iteration": list(range(1, len(initial_xs) + 1))}
                    new_data["Objective"] = [-y if maximize_objective else y for y in initial_ys]
                    for i, name in enumerate(param_names):
                        new_data[name] = [x[i] for x in initial_xs]
                    experiments_source.stream(new_data)

                process_and_plot_latest_results()
                update_initial_points_warning(len(experiment_history))
                update_status("🟢 Ready. Click 'Suggest Next Experiment' to begin.")
                set_ui_state(phase='optimization')

            doc.add_next_tick_callback(final_callback)

        except Exception as e:
            # Pass the exception 'e' to the error_callback
            doc.add_next_tick_callback(partial(error_callback, e))

    threading.Thread(target=worker, args=(config,)).start()

# Modified error_callback to accept the exception as an argument
def error_callback(e):
    """Updates the status div with an error message."""
    update_status(f"❌ Error during worker execution: {e}", is_error=True)
    set_ui_state(phase='setup')


def suggest_next_experiment():
    """Asks the optimizer for the next point and displays it with predictions."""
    global suggested_x

    try:
        if optimizer is None:
            update_status("❌ Error: Optimizer is not initialized. Please reset and complete setup.", is_error=True)
            return

        update_status("🤔 Thinking of the next best experiment...")

        suggested_x = optimizer.ask()

        expected_internal, uncertainty_val = 0.0, 1.0

        if optimizer.models:
            model = optimizer.models[-1]
            X_transformed = optimizer.space.transform([suggested_x])

            if isinstance(model, GaussianProcessRegressor):
                mean, std = model.predict(X_transformed, return_std=True)
                expected_internal, uncertainty_val = mean[0], std[0]
            elif isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
                predictions = [tree.predict(X_transformed)[0] for tree in model.estimators_]
                expected_internal, uncertainty_val = np.mean(predictions), np.std(predictions)

        elif experiment_history:
            ys = [item[1] for item in experiment_history]
            expected_internal = np.mean(ys)
            uncertainty_val = np.std(ys) if len(ys) > 1 else 1.0

        expected_display_val = -expected_internal if maximize_objective else expected_internal
        expected_display_str = f"{expected_display_val:.4f}"
        uncertainty_str = f"{uncertainty_val:.4f}"
        
        # New: Clear direction-specific prediction label
        direction_label = "Maximum" if maximize_objective else "Minimum"

        suggestion_html = f"<h5>💡 Suggested Experiment (Iteration {len(experiment_history)+1}):</h5>"
        suggestion_html += "<ul>"
        for name, val in zip(param_names, suggested_x):
            suggestion_html += f"<li><b>{name}:</b> {val:.4f}</li>"
        suggestion_html += "</ul>"

        suggestion_html += f"<b>Predicted {direction_label}:</b> {expected_display_str}<br>"
        suggestion_html += f"<b>Model Uncertainty:</b> {uncertainty_str}"

        suggestion_div.text = suggestion_html
        actual_result_input.disabled = False
        submit_result_button.disabled = False
        update_status("💡 Suggestion received. Please provide the result.")

    except Exception as e:
        import traceback
        print("--- ERROR IN suggest_next_experiment ---")
        traceback.print_exc()
        print("--------------------------------------")
        update_status(f"❌ An error occurred while suggesting an experiment: {e}", is_error=True)


def submit_result():
    """Submits the experimental result to the optimizer and updates the display."""
    global suggested_x
    result_value = actual_result_input.value
    if result_value is None or suggested_x is None: return

    update_status("🔄 Updating model with new result...")

    internal_result = -result_value if maximize_objective else result_value
    optimizer.tell(suggested_x, internal_result)
    experiment_history.append((suggested_x, internal_result))

    new_data = {"Iteration": [len(experiment_history)], "Objective": [result_value]}
    for i, name in enumerate(param_names):
        new_data[name] = [suggested_x[i]]
    experiments_source.stream(new_data)

    process_and_plot_latest_results()
    update_initial_points_warning(len(experiment_history))

    suggestion_div.text, actual_result_input.value, suggested_x = "", None, None
    actual_result_input.disabled, submit_result_button.disabled = True, True
    update_status("✅ Model updated. Ready for the next suggestion.")

def process_and_plot_latest_results():
    """Finds the best result from history and updates plots and summary stats."""
    if not experiment_history:
        best_result_div.text = ""
        return

    # Find the best result based on the internal (minimized) objective value
    best_idx = 0
    best_y_internal = float('inf')
    for idx, (_, y_val) in enumerate(experiment_history):
        if y_val < best_y_internal:
            best_y_internal = y_val
            best_idx = idx

    best_x = experiment_history[best_idx][0]
    best_y_display = -best_y_internal if maximize_objective else best_y_internal

    # Check if we've reached the tolerance threshold
    tolerance_reached = False
    if not maximize_objective and best_y_display < TOLERANCE:
        tolerance_reached = True
        update_status(f"✅ Optimum reached! (Value < {TOLERANCE:.0e})", is_error=False)

    results_html = f"<h3>Current Best: {best_y_display:.6f}</h3><ul>"
    for name, value in zip(param_names, best_x):
        results_html += f"<li><b>{name}:</b> {value:.6f}</li>"
    results_html += "</ul>"
    
    if tolerance_reached:
        results_html += f"<div style='color:green; font-weight:bold;'>✓ Optimum reached (value < {TOLERANCE:.0e})</div>"
        
    best_result_div.text = results_html

    # More efficient convergence tracking
    iters = list(range(1, len(experiment_history) + 1))
    best_values_so_far = []
    current_best = float('inf')
    for _, y_val in experiment_history:
        if y_val < current_best:
            current_best = y_val
        display_val = -current_best if maximize_objective else current_best
        best_values_so_far.append(display_val)

    convergence_source.data = {'iter': iters, 'best_value': best_values_so_far}

def reset_all():
    """Resets the entire application state."""
    global optimizer, param_names, dimensions, experiment_history, suggested_x, initial_loaded_data
    optimizer, param_names, dimensions, experiment_history, suggested_x = None, [], [], [], None
    initial_loaded_data = [] # Clear the loaded data on reset

    experiments_source.data = dict(Iteration=[], Objective=[])
    data_table.columns = [TableColumn(field="Iteration", title="Iteration"), TableColumn(field="Objective", title="Objective")]
    convergence_source.data = dict(iter=[], best_value=[])

    suggestion_div.text, best_result_div.text = "", ""
    actual_result_input.value = None
    
    update_initial_points_warning(0)
    update_direction_indicator()

    objective_header.text = f"<b>{objective_name_input.value or 'Objective'}</b>"
    for i in range(MAX_PARAMS):
        initial_data_headers[i].text = f"<b>{param_name_inputs[i].value or f'Param {i+1}'}</b>"

    update_status("🟢 Ready. Define optimization problem.")
    set_ui_state(phase='setup')
    update_initial_random_points_info()
    # Ensure initial data layout is hidden on reset
    initial_data_spinner.value = 0 # Reset spinner value
    initial_data_header_row.visible = False
    for row_widget in initial_data_rows:
        row_widget.visible = False


def update_status(message, is_error=False):
    """Updates the status div with a message and optional error styling."""
    status_div.text = message
    if is_error:
        status_div.styles = {
            'background-color': '#F8D7DA',
            'color': '#721C24',
            'border': '1px solid #F5C6CB',
            'padding': '10px',
            'border-radius': '5px'
        }
    else:
        status_div.styles = {
            'background-color': 'transparent',
            'color': 'black',
            'border': 'none'
        }

def handle_csv_upload(attr, old, new):
    """
    Handles the uploaded CSV file, parses it, and populates the initial data fields.
    """
    global initial_loaded_data # Access the global variable
    if not new:
        print("DEBUG: No new CSV file selected. Clearing initial data display.")
        update_status("No CSV file selected.", is_error=False)
        initial_loaded_data = [] # Clear global data
        initial_data_spinner.value = 0
        doc.add_next_tick_callback(partial(on_initial_data_change, 'value', None, 0)) # Ensure UI hides
        return

    try:
        # Decode base64 content to string, handling BOM
        decoded_csv_content = base64.b64decode(new).decode('utf-8-sig')
        csv_data = io.StringIO(decoded_csv_content)
        reader = csv.reader(csv_data)
        
        header = next(reader, None)
        if not header:
            raise ValueError("CSV file is empty or missing header row.")

        # Assume first column is objective, rest are parameters
        obj_col_name = header[0].strip()
        csv_param_names = [name.strip() for name in header[1:]]

        print(f"DEBUG: CSV Header - Objective: '{obj_col_name}', Parameters: {csv_param_names}")

        # Validate against current setup
        current_num_params = num_params_spinner.value
        current_param_names = [param_name_inputs[i].value.strip() for i in range(current_num_params)]
        current_obj_name = objective_name_input.value.strip()

        print(f"DEBUG: Current UI Setup - Objective: '{current_obj_name}', Parameters: {current_param_names}")


        if obj_col_name.lower() != current_obj_name.lower(): # Case-insensitive check
            update_status(f"CSV header objective '{obj_col_name}' does not match '{current_obj_name}'. Please ensure objective name matches (case-insensitive).", is_error=True)
            return
        
        if len(csv_param_names) != current_num_params:
            update_status(f"CSV has {len(csv_param_names)} parameters, but current setup expects {current_num_params}. Please adjust 'Number of Input Parameters' or CSV.", is_error=True)
            return

        # Check if parameter names match (case-insensitive)
        if not all(p_csv.lower() == p_setup.lower() for p_csv, p_setup in zip(csv_param_names, current_param_names)):
            update_status("CSV parameter names do not match current setup. Please ensure order and names are identical (case-insensitive).", is_error=True)
            return

        temp_parsed_data = [] # Use a temporary list
        for i, row_data in enumerate(reader):
            if not row_data:
                continue # Skip empty rows
            if len(row_data) != len(header):
                update_status(f"Row {i+2} has incorrect number of columns. Expected {len(header)}, got {len(row_data)}.", is_error=True)
                return

            try:
                obj_value = float(row_data[0])
                param_values = [float(v) for v in row_data[1:]]
                temp_parsed_data.append({"x": param_values, "y": obj_value})
            except ValueError:
                update_status(f"Row {i+2} contains non-numeric data. All data values must be numbers.", is_error=True)
                return
        
        if not temp_parsed_data:
            update_status("No valid data rows found in CSV.", is_error=True)
            return

        if len(temp_parsed_data) > MAX_INITIAL_POINTS:
            update_status(f"CSV contains {len(temp_parsed_data)} data points, but only the first {MAX_INITIAL_POINTS} will be loaded due to UI limits.", is_error=False)
            temp_parsed_data = temp_parsed_data[:MAX_INITIAL_POINTS]

        print(f"DEBUG: Parsed {len(temp_parsed_data)} data points from CSV.")
        initial_loaded_data = temp_parsed_data # Assign to global variable

        # First, reset all initial data input fields to None to clear previous values
        for i in range(MAX_INITIAL_POINTS):
            initial_objective_inputs[i].value = None
            for j in range(MAX_PARAMS):
                initial_param_inputs[i][j].value = None

        # Populate the individual spinners with parsed data for visual feedback
        for i, data_point in enumerate(initial_loaded_data): # Use initial_loaded_data
            initial_objective_inputs[i].value = data_point["y"]
            for j, val in enumerate(data_point["x"]):
                initial_param_inputs[i][j].value = val
        
        # Set the spinner value last, which should trigger on_initial_data_change
        # This order ensures the values are present when visibility is updated.
        initial_data_spinner.value = len(initial_loaded_data)

        # Explicitly call on_initial_data_change to ensure UI elements are visible
        # Use doc.add_next_tick_callback to ensure this runs after values are set.
        doc.add_next_tick_callback(partial(on_initial_data_change, 'value', None, initial_data_spinner.value))

        update_status(f"✅ Successfully loaded {len(initial_loaded_data)} data points from CSV.", is_error=False)
        
        # Removed: csv_file_input.value = None as it's a readonly property

    except Exception as e:
        import traceback
        print("--- ERROR IN handle_csv_upload ---")
        traceback.print_exc()
        print("----------------------------------")
        update_status(f"❌ Error processing CSV: {e}", is_error=True)


# --- UI Widget Definitions ---
setup_title = Div(text="<h2>1. Define Optimization Problem</h2>")
num_params_spinner = Spinner(title="Number of Input Parameters", low=1, high=20, step=1, value=2, width=200)
objective_name_input = TextInput(title="Objective Name (e.g., Yield, Purity):", value="Objective")
objective_type_select = RadioButtonGroup(labels=["Maximize", "Minimize"], active=0)

# NEW: Direction indicator next to objective name
direction_indicator = Div(text="(Maximizing)", styles={
    'color': 'green',
    'font-style': 'italic',
    'font-weight': 'bold',
    'margin-left': '10px'
})

MAX_PARAMS = 20
param_rows, param_name_inputs, param_low_spinners, param_high_spinners = [], [], [], []
initial_data_headers = []

for i in range(MAX_PARAMS):
    name_input = TextInput(value=f"Parameter {i+1}", width=150)
    low_spinner = Spinner(title="Min", value=10, step=1, low=None, high=None, width=100)
    high_spinner = Spinner(title="Max", value=50, step=1, low=None, high=None, width=100)

    param_name_inputs.append(name_input)
    param_low_spinners.append(low_spinner)
    param_high_spinners.append(high_spinner)
    param_rows.append(row(name_input, low_spinner, high_spinner, sizing_mode="stretch_width", visible=(i < num_params_spinner.value)))

    header_div = Div(text=f"<b>Parameter {i+1}</b>", width=80, visible=(i < num_params_spinner.value), styles={'text-align': 'center'})
    initial_data_headers.append(header_div)

    name_input.on_change('value', partial(on_param_name_change, i))
    
    low_spinner.on_change('value', partial(on_param_range_change, i))
    high_spinner.on_change('value', partial(on_param_range_change, i))


initial_data_title = Div(text="<h4>Enter Existing Experimental Data (Recommended)</h4>")
initial_data_spinner = Spinner(title="Number of existing data points:", low=0, high=10, step=1, value=0, width=200)

# NEW: CSV File Input
csv_file_input = FileInput(
    accept=".csv",
    multiple=False,
    title="Upload CSV Data",
    width=200
)

warning_text = """
<p style="color: #856404; background-color: #FFF3CD; border: 1px solid #FFEEBA; padding: 10px; border-radius: 5px;">
⚠️ <b>Note:</b> It is recommended to have at least five entered data points. Model performance may be poor with fewer points.
</p>
"""
initial_points_warning_div = Div(text=warning_text, visible=False)

# NEW: Div to inform about initial random points strategy
initial_random_points_info_div = Div(text="", styles={'margin-top': '10px'})


objective_header = Div(text=f"<b>{objective_name_input.value}</b>", width=80, styles={'text-align': 'center'})
initial_data_header_row = row(objective_header, *initial_data_headers, sizing_mode="stretch_width", visible=False)

MAX_INITIAL_POINTS = 10
initial_data_rows, initial_param_inputs, initial_objective_inputs = [], [], []
for i in range(MAX_INITIAL_POINTS):
    default_low = param_low_spinners[0].value
    default_high = param_high_spinners[0].value
    param_inputs = [Spinner(width=80, step=0.01, value=None, low=default_low, high=default_high, visible=(j < num_params_spinner.value)) for j in range(MAX_PARAMS)]
    obj_input = Spinner(width=80, step=0.01, value=None)
    initial_param_inputs.append(param_inputs)
    initial_objective_inputs.append(obj_input)
    initial_data_rows.append(row(obj_input, *param_inputs, sizing_mode="stretch_width", visible=False))

model_title = Div(text="<h4>Model Configuration</h4>")
surrogate_select = Select(title="Surrogate Model:", value="GP", options=["GP", "RF", "ET"])
acq_func_select = Select(title="Acquisition Function:", value="gp_hedge", options=["gp_hedge", "EI", "PI", "LCB"])
lock_setup_button = Button(label="Lock Setup & Start Optimization", button_type="primary", width=400)

workflow_title = Div(text="<h2>2. Run Optimization Workflow</h2>")
suggest_button = Button(label="Suggest Next Experiment", button_type="success", width=400)
suggestion_div = Div()
actual_result_input = Spinner(title="Enter Measured Objective Value:", value=None, step=0.01)
submit_result_button = Button(label="Submit Result & Update Model", button_type="warning", width=400)

status_div = Div(text="🟢 Ready. Define optimization problem.")
best_result_div = Div()
data_table = DataTable(source=experiments_source, columns=[TableColumn(field="Iteration", title="Iteration"), TableColumn(field="Objective", title="Objective")], width=600, height=200, editable=False)
p_conv = figure(height=300, width=600, title="Convergence Plot", x_axis_label="Iteration", y_axis_label="Best Objective Value")
p_conv.line(x='iter', y='best_value', source=convergence_source, line_width=2)
reset_button = Button(label="Reset Experiment", button_type="danger", width=400)

initial_data_layout = column(
    initial_data_title,
    row(initial_data_spinner, csv_file_input), # Placed CSV input next to spinner
    initial_points_warning_div,
    initial_random_points_info_div,
    initial_data_header_row,
    *initial_data_rows
)

setup_widgets = [
    num_params_spinner, objective_name_input, objective_type_select,
    surrogate_select, acq_func_select, lock_setup_button, initial_data_spinner,
    csv_file_input # Added to setup widgets for state management
] + param_name_inputs + param_low_spinners + param_high_spinners + [
    item for sublist in initial_param_inputs for item in sublist
] + initial_objective_inputs

optimization_widgets = [suggest_button, actual_result_input, submit_result_button]
all_buttons = [lock_setup_button, suggest_button, submit_result_button, reset_button]

# --- Attach Callbacks ---
objective_name_input.on_change('value', on_objective_name_change)
num_params_spinner.on_change('value', on_num_params_change)
initial_data_spinner.on_change('value', on_initial_data_change)
surrogate_select.on_change('value', on_surrogate_model_change)
csv_file_input.on_change('value', handle_csv_upload) # Attach CSV upload handler

# NEW: Direction indicator callback
def on_objective_type_change(attr, old, new):
    global maximize_objective
    maximize_objective = (new == 0)
    update_direction_indicator()
objective_type_select.on_change('active', on_objective_type_change)

lock_setup_button.on_click(lock_in_setup)
suggest_button.on_click(suggest_next_experiment)
submit_result_button.on_click(submit_result)
reset_button.on_click(reset_all)

# Combine objective name and direction indicator
objective_row = row(objective_name_input, direction_indicator)

controls_col = column(
    setup_title, num_params_spinner, *param_rows,
    Div(text="<b>Objective Configuration:</b>"),
    objective_row, objective_type_select,
    initial_data_layout,
    model_title, row(surrogate_select, acq_func_select),
    lock_setup_button, workflow_title,
    suggest_button, suggestion_div, actual_result_input, submit_result_button,
    reset_button, status_div,
    width=500
)
results_col = column(best_result_div, data_table, p_conv)
main_layout = row(controls_col, results_col)

doc.add_root(main_layout)

# --- Document Ready Handler ---
def on_doc_ready(event):
    set_ui_state(phase='setup')
    update_status("🟢 Ready. Define optimization problem.")
    for i in range(num_params_spinner.value):
        on_param_range_change(i, None, None, None)
    update_initial_random_points_info()
    update_direction_indicator()
    # Ensure initial data layout is hidden on initial load
    initial_data_spinner.value = 0 # Reset spinner value
    initial_data_header_row.visible = False
    for row_widget in initial_data_rows:
        row_widget.visible = False

doc.on_event("document_ready", on_doc_ready)
