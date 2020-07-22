import openmdao.api as om
from autograd import grad
import numpy as np
from collections import defaultdict

def array_to_scalar(out):
    if out.shape[0] == 1:
        return out[0]
    return out

def find_output_eq(eqs, output):
    for eqname, eq in eqs.items():
        if eq.output_name == output:
            return eqname, eq
    return False

def find_input_eqs(eqs, inputn):
    return [(eqname, eq) for eqname, eq in eqs.items() 
        if inputn in eq.input_names]

def args_in_order(name_dict, names):
    return [name_dict[in_var] for in_var in names]

class Equation():
    def __init__(self, output_name, lambda_fx, input_names=None):
        self.input_names = (input_names if input_names 
            else lambda_fx.__code__.co_varnames)
        self.output_name = output_name
        self.fx = lambda_fx
        self.jfx = grad(lambda x: lambda_fx(*x))
        
    def evaldict(self, indict):
        return self.fx(*args_in_order(indict, self.input_names))
        
class Expcomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('equation')
        
    def setup(self):
        equation = self.options['equation']
        for name in equation.input_names:
            self.add_input(name)
        self.add_output(equation.output_name)
        self.declare_partials(equation.output_name, equation.input_names)
            
    def compute(self, inputs, outputs):
        equation = self.options['equation']
        outputs[equation.output_name] = equation.evaldict(inputs)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        eq = self.options['equation']
        jfx = eq.jfx
        args = np.array(args_in_order(inputs, eq.input_names)).T[0]
        J = jfx(args)
        for idx, input_name in enumerate(eq.input_names):
            partials[eq.output_name, input_name] = J[idx]

class EqComp(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('lhseq')
        self.options.declare('rhseq')
        self.options.declare('solvefor')

    def setup(self):
        solvefor = self.options['solvefor']
        lhseq, rhseq = self.options['lhseq'], self.options['rhseq']
        for name in lhseq.input_names+rhseq.input_names:
            if name != solvefor:
                self.add_input(name)
        self.add_output(solvefor) 
    
    def apply_nonlinear(self, inputs, outputs, residuals):
        solvefor = self.options['solvefor']
        allvars = dict(**inputs, **outputs)
        lhseq, rhseq = self.options['lhseq'], self.options['rhseq']
        out = lhseq.evaldict(allvars)-rhseq.evaldict(allvars)
        residuals[solvefor] = out

class quickModel():
    def __init__(self):
        self.neqs = 0
        self.prob = om.Problem()
        self.var_names = set()
        self.output_dict = defaultdict(list) #this is needed for equality check
        self.eqs = dict()
        self.params = self.prob.model.add_subsystem('p', om.IndepVarComp(),
            promotes=["*"])
        self.eqgroup = self.prob.model.add_subsystem('eqs', om.Group(), 
            promotes=['*'])
        self.eqgroup.nonlinear_solver = nl = om.NewtonSolver(
                solve_subsystems=False, maxiter=100)
        self.eqgroup.linear_solver = om.DirectSolver()
        #nl.options['iprint'] = 2
        #nl.options['debug_print'] = True
        self.prob.set_solver_print(level=2)

        self.setup_done = False
        
    def set_var(self, name, value):
        if name not in self.var_names:
            self.var_names.add(name)
            self.params.add_output(name, value)
            self.output_dict[name].append('params')
        else:
            self.prob[name] = value

    def set_equal(self, lhs_var, rhs_var, solvefor, solveforval=1.):
        lhname, lhseq = find_output_eq(self.eqs, lhs_var)
        rhname, rhseq = find_output_eq(self.eqs, rhs_var)
        model = self.prob.model
        #eq = EqComp(solvefor=solvefor, lhseq=lhseq, rhseq=rhseq)
        
        equality_name = 'h_{}_{}_{}'.format(solvefor, lhs_var, rhs_var)
        bal = om.BalanceComp(solvefor, val=solveforval, rhs_val=solveforval,
            lhs_name=lhs_var, rhs_name=rhs_var)
        self.eqgroup.add_subsystem(name=equality_name, subsys=bal)

        #for inp_eq_name,eq in find_input_eqs(self.eqs, solvefor):
        model.connect('{}.{}'.format(equality_name, solvefor), 
                '{}'.format(solvefor))
        model.connect('{}'.format(lhs_var), 
            '{}.{}'.format(equality_name, lhs_var))
        model.connect('{}'.format(rhs_var), 
            '{}.{}'.format(equality_name, rhs_var))

    def add_eq(self, output_name, lambda_fx, input_names=None):
        self.neqs += 1
        eq = Equation(output_name, lambda_fx, input_names)
        eqname = 'eq{}'.format(self.neqs)
        self.eqs[eqname] = eq
        self.var_names.union({elt for elt in (output_name,)+eq.input_names}) 
        
        output_promotes = [eq.output_name]

        # Check if the equation is already an output
        # if self.output_dict[output_name]:
        #     # if it is, make sure to add an equality constraint
        #     most_recent_output = self.output_dict[output_name][-1]
        #     self.set_equal(output_name, most_recent_output, eqname)
        #     output_promotes = []
        # # Associate the variable with the output of this equation
        # self.output_dict[output_name].append(eqname)

        # This is the key part of this equation
        self.eqgroup.add_subsystem(eqname, 
            Expcomp(equation=eq),
            promotes_outputs=output_promotes,  # only promotes if no equality
            promotes_inputs=eq.input_names)


    def get_rhs(self, eqname):
        ins = {info['prom_name']:info['value'] 
            for key, info in getattr(self.eqgroup, eqname).list_inputs(
                    prom_name=True, out_stream=None)}

        out = array_to_scalar(self.eqs[eqname].evaldict(ins))
        return out
    
    def getval(self, var, get_both=False):
        ins = {info['prom_name']:info['value'] for key, info in 
                self.prob.model.list_inputs(prom_name=True, out_stream=None)}
        outs = {info['prom_name']:info['value'] for key, info in 
                self.prob.model.list_outputs(prom_name=True, out_stream=None)}
        out = (array_to_scalar(ins[var]),) if var in ins else ()
        out += (array_to_scalar(outs[var]),) if var in outs else ()
        if get_both:
            return out
        return out[0]

    def run(self):
        #if not self.setup:
        self.prob.setup()
        #self.setup = True
        self.prob.run_model()
        self.prob.model.list_inputs()
        self.prob.model.list_outputs()