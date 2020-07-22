import sympy as sp
from sympy.abc import a
import autograd.numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()

class Var(sp.core.Symbol):
    def __new__(cls, name, value=None, unit=None):
        out = super().__new__(cls, name)
        out.varval = value
        out.varunit = unit
        return out

def get_unit_multi(unit):
    return unit.to_base_units().magnitude

def var_generator(m):
    def wrapped(name, value=None, unit=None):
        unit = ureg(unit)
        if isinstance(value, sp.Basic):
            free_symbols = list(value.free_symbols)
            rhs = value
            rhsfx = sp.lambdify(free_symbols, rhs, modules=np)
            rhs_unit = rhsfx(*(free_symbol.varunit 
                for free_symbol in free_symbols))
            convert = np.array([get_unit_multi(free_symbol.varunit) for 
                    free_symbol in free_symbols])
            if unit:
                assert(unit.dimensionality == rhs_unit.dimensionality)
                conversion_unit = unit
            else:
                conversion_unit = ureg.Quantity(1, rhs_unit.units)

            factor = get_unit_multi(conversion_unit)

            rhsfx_corrected = lambda *args: rhsfx(*(
                convert*np.array(args).flatten()))/factor
            m.add_eq(name, rhsfx_corrected, tuple([free_symbol.name 
                for free_symbol in free_symbols]))
            return Var(name, unit=unit)
        else:
            if value:
                if unit == ureg('deg'):
                    value = (value*unit).to(ureg('rad')).magnitude
                    unit = ureg('rad')
                m.set_var(name, value)
            return Var(name, value, unit)
    return wrapped
    
