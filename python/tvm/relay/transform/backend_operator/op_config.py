from pathlib import Path
from .utils import extract_attrs, get_data_shape
import pickle

cur_dir_path = Path(__file__).parent.absolute()
COST_LOG = f"{cur_dir_path}/logs/operator_cost.log"

# configuration includes operator name, operator type (backend operators from different targets might have the same type),
# data shape of all free variables, and node attributes
class Config(object):
  # We have data_shape and attrs as arguments for debugging purpose
  def __init__(self, op_name, op_type, expr, data_shape=None, attrs=None):
    self._op_name = op_name
    self._op_type = op_type
    
    if expr != None:
      self._data_shape = get_data_shape(expr)
      self._attrs = extract_attrs(expr)
    else:
      # Debugging purpose
      self._data_shape = data_shape
      self._attrs = attrs

  def __hash__(self):
    return hash((self._op_name, self._op_type, self._data_shape, self._attrs))

  def __eq__(self, other):
#     print(f"Check equality, {type(self._op_name)}, {type(self._op_type)}, {type(self._data_shape)}, {type(self._attrs)}")
    return (self._op_name == other._op_name and self._op_type == other._op_type 
    and self._data_shape == other._data_shape and self._attrs == other._attrs)

  def __repr__(self):
    return "op_name: {0}, op_type: {1}, data_shape: {2}, attrs: {3}".format(
      self._op_name, self._op_type, self._data_shape, self._attrs)

# class to save costs of already evaluated configurations so we do not need to reevaluate them
class MeasuredConfigs(object):
  def __init__(self):
    # maps configurations already measured to the measured cost (in ms)
    self.measured_configs = dict()

  def get_cost(self, config):
    if config in self.measured_configs:
      return self.measured_configs[config]
    return None

  # cost is (mean(cost), std(cost))
  def save_cost(self, config, cost):
    self.measured_configs[config] = cost

  def save_to_log(self):
    with open(COST_LOG, 'wb+') as log:
      pickle.dump(self.measured_configs, log)

  def load_from_log(self):
    try:
      with open(COST_LOG, 'rb') as log:
        self.measured_configs = pickle.load(log)
    except:
#       pass
      raise Exception(f'{COST_LOG} is not valid')