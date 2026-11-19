"""Author generation AST retained, deterministic native7 prep removed.

Original graph Louvain is retained solely for its inherited NumPy RNG effect.
The matrix-completion observation raises GenerationCaptured before native7.
"""
import ast,copy,hashlib,inspect,textwrap,types

class GenerationCaptured(Exception):
    """One formal final matrix was saved; skip all following native evaluation."""

# Exact local targets belonging solely to evaluation in the four author mains.
PREP_TARGETS=frozenset((
 'mat0_degree','mat0_deg_dist','mat0_evc','mat0_evc_a','mat0_evc_ak',
 'mat0_evc_val','evc_kn','mat0_diam','mat0_cc','mat0_mod'))

def dump(node):return ast.dump(node,include_attributes=False)
def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generation_only_function(function):
    source=textwrap.dedent(inspect.getsource(function))
    tree=ast.parse(source);original=copy.deepcopy(tree);removed=[]
    endpoint=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='mat2_graph' for t in n.targets)]
    assert len(endpoint)==1 and isinstance(endpoint[0].value,ast.Call)
    call=endpoint[0].value
    assert isinstance(call.func,ast.Attribute) and call.func.attr=='from_numpy_array'
    assert isinstance(call.args[0],ast.Name) and call.args[0].id=='mat2'
    cutoff=endpoint[0].lineno
    before_nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and n.lineno<cutoff and
        any(isinstance(t,ast.Name) and t.id in PREP_TARGETS for t in n.targets)]
    assert {t.id for n in before_nodes for t in n.targets if isinstance(t,ast.Name)}==PREP_TARGETS
    # Every removed value is dead with respect to all code up to final mat2.
    # References from other removed evaluation preparations are allowed.
    class RemovePrep(ast.NodeTransformer):
        def visit_Assign(self,node):
            if node.lineno<cutoff and any(isinstance(t,ast.Name) and t.id in PREP_TARGETS for t in node.targets):
                removed.append(dict(line=node.lineno,source=ast.get_source_segment(source,node),ast=dump(node)))
                return None
            return self.generic_visit(node)
    transformed=RemovePrep().visit(tree);ast.fix_missing_locations(transformed)
    remaining_prefix=[n for n in ast.walk(transformed) if getattr(n,'lineno',cutoff+1)<cutoff]
    assert not [n.id for n in remaining_prefix if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in PREP_TARGETS]
    original_remaining=RemovePrep().visit(copy.deepcopy(original));ast.fix_missing_locations(original_remaining)
    assert dump(original_remaining)==dump(transformed)
    partitions=[n for n in ast.walk(transformed) if isinstance(n,ast.Assign) and getattr(n,'lineno',cutoff+1)<cutoff and any(isinstance(t,ast.Name) and t.id=='mat0_par' for t in n.targets)]
    assert len(partitions)==1 and isinstance(partitions[0].value,ast.Call)
    assert isinstance(partitions[0].value.func,ast.Attribute) and partitions[0].value.func.attr=='best_partition'
    module_code=compile(transformed,function.__code__.co_filename,'exec')
    fn_code=next(c for c in module_code.co_consts if isinstance(c,types.CodeType) and c.co_name==function.__name__)
    new=types.FunctionType(fn_code,function.__globals__,function.__name__,function.__defaults__,function.__closure__)
    new.__kwdefaults__=function.__kwdefaults__
    report=dict(version='native7_skip_v1',function=function.__name__,original_function_source_sha256=sha(source),
        removed_deterministic_metric_prep=removed[:len(before_nodes)],final_matrix_boundary_line=cutoff,
        generation_ast_identical_except_dead_metric_prep=True,removed_names_not_read_by_generation=True,
        original_louvain_kept_for_rng_compatibility=True,parameter_defaults_unchanged=True,
        synthetic_native7_not_executed=True,original_native7_not_calculated=True,
        final_graph_conversion_skipped_after_matrix_capture=True)
    return new,report

