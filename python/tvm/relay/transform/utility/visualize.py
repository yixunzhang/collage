import tvm
import tvm.relay as relay
import tvm.relay.testing as testing
from tvm.relay.transform.utility.debug_helper import printe
from tvm.relay.transform.backend_operator.utils import *

from graphviz import Digraph

def _traverse_expr(node, node_dict):
    if node in node_dict:
        return
    # if isinstance(node, relay.op.op.Op):
    #    return
    if isinstance(node, tvm.ir.op.Op):
        return

    # print("{} : {}".format(node, type(node)))
    node_dict[node] = len(node_dict)

def get_node_color(node):
    backend_name = get_backend_from_backend_op_annotation(node.backend)

    # If this is default (no backend op assignment)
    color = "ivory"

    if backend_name == "tensorrt":
        color = "orange"
    elif backend_name[:3] == "tvm":
        color = "greenyellow"
    elif backend_name[:5] == "cudnn":
        color = "yellow"
    elif backend_name[:6] == "cublas":
        color = "grey60"

    return color

def visualize_network(expr, file_name):

    dot = Digraph(format='pdf')
    dot.attr(rankdir='BT')
    # dot.attr('node', shape='box')

    node_dict = {}
    relay.analysis.post_order_visit(expr, lambda node: _traverse_expr(node, node_dict))

    for node, node_idx in node_dict.items():
        node_idx_backend_str = f"[{node_idx}, {node.backend}]"
        node_color = get_node_color(node)

        if isinstance(node, relay.Function):
            # elif isinstance(node, relay.expr.Function):
            dot.node(str(node_idx), f'Function ({node_idx})', shape='doubleoctagon')
            dot.edge(str(node_dict[node.body]), str(node_idx))

        elif isinstance(node, relay.expr.Var):
            if isinstance(node.type_annotation, tvm.ir.type.TupleType):
                type_info = node.type_annotation.fields
                tensor_info = f'Tensor[TupleType{tuple(type_info)}]'
            else:
                type_info = node.type_annotation.shape
                tensor_info = f'Tensor[{tuple(type_info)}, {node.type_annotation.dtype}]'

            dot.node(str(node_idx), \
                     f'{node.name_hint} {node_idx_backend_str}:\n{tensor_info}', \
                     shape='rectangle'
                     )
        elif isinstance(node, relay.expr.GlobalVar):
            dot.node(str(node_idx), \
                     f'{node.name_hint} {node_idx_backend_str}', \
                     shape='rectangle'
                     )

        elif isinstance(node, relay.Constant):
            dot.node(str(node_idx), \
                     f'Constant {node_idx_backend_str}:\nTensor[{tuple(node.data.shape)}, {node.data.dtype}]', \
                     shape='rectangle'
                     )

        elif isinstance(node, relay.expr.Call):
            args = [node_dict[arg] for arg in node.args]
            # dot.node(str(node_idx), f'Call(op={node.op.name})')
            if isinstance(node.op, tvm.relay.Function):
                dot.node(str(node_idx), f'Call {node_idx_backend_str}(Function({node_dict[node.op.body]}))', shape='ellipse',
                         style='filled', color=node_color)
            else:
                if isinstance(node.op, relay.expr.GlobalVar):
                    dot.node(str(node_idx), f'Call{node_idx_backend_str}(GlobalVar={node.op.name_hint})', shape='ellipse', style='filled', color=node_color)
                else:
                    dot.node(str(node_idx), f'Call {node_idx_backend_str}(op={node.op.name})', shape='ellipse', style='filled', color=node_color)


            for arg in args:
                dot.edge(str(arg), str(node_idx))
        elif isinstance(node, relay.expr.TupleGetItem):
            dot.node(str(node_idx), f'TupleGetItem {node_idx_backend_str}(idx={node.index})', shape='ellipse', style='filled', color=node_color)
            dot.edge(str(node_dict[node.tuple_value]), str(node_idx))
        elif isinstance(node, relay.expr.Tuple):
            args = [node_dict[field] for field in node.fields]
            dot.node(str(node_idx), f'Tuple {node_idx_backend_str}(fileds=none)', shape='ellipse', style='filled', color=node_color)
            for arg in args:
                dot.edge(str(arg), str(node_idx))
        else:
            raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

    dot.render(f'analysis/results/net_figs/{file_name}.gv')

