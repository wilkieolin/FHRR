import jax.numpy as jnp
from jax import random, vmap, jit

#create a helper function to generate random symbols
def generate_symbols(key, number: int, dimensionality: int):
    return random.uniform(key, minval=-1.0, maxval=1.0, shape=(number, dimensionality))

@jit
#similarity of FHRR vectors is defined by the average cosine of the difference between each angle in the two symbols being compared.
def similarity(a,b):
    assert a.shape[-1] == b.shape[-1], "VSA Dimension must match: " + str(a.shape) + " " + str(b.shape)
    #multiply values by pi to move from (-1, 1) to (-π, π)
    pi = jnp.pi
    a = jnp.multiply(a, pi)
    b = jnp.multiply(b, pi)
    #calculate the mean cosine similarity between the vectors
    similarity = jnp.mean(jnp.cos(a - b), axis=1)
    return similarity

#given two sets of symbols, measure the similarity between each pair of inputs
def similarity_outer(a,b):
    assert a.shape[1] == b.shape[1], "VSA Dimension must match: " + str(a.shape) + " " + str(b.shape)
    sim_op = lambda x: similarity(x, b)
    
    return vmap(sim_op)(a)

#shift angles (radian-normalized) from (-inf, inf) to (-1, 1)
def remap_phase(x):
    x = jnp.mod(x, 2.0)
    x = -2.0 * jnp.greater(x, 1.0) + x

    return x

#Convert our phasor symbols on the domain (-1, 1) to complex numbers
# on the unit circle
def unitary_to_cmpx(symbols):
    #convert each angle to a complex number
    pi = jnp.pi
    j = jnp.array([0+1j])
    #sum the complex numbers to find the bundled vector
    cmpx = jnp.exp(pi * j * symbols)

    return cmpx

#Convert complex numbers back to phasor symbols on the domain (-1, 1)
def cmpx_to_unitary(cmpx):
    pi = jnp.pi
    #convert the complex sum back to an angle
    symbol = jnp.angle(cmpx) / pi

    return symbol

#bundle a list of symbols together
def bundle_list(*symbols):
    #stack the vararg inputs into an array
    symbols = jnp.stack(symbols, axis=0)

    return bundle(symbols)

#Bundling operation for FHRR, the weight of the first symbol
# can be scaled with 'n'
def bundle(symbols, n=-1):
    #sum the complex numbers to find the bundled vector
    cmpx = unitary_to_cmpx(symbols)
    if n > 1:
        cmpx_0 = n * cmpx[0:1, :]
        cmpx_1 = cmpx[1:, :]
        cmpx = jnp.stack((cmpx_0, cmpx_1), axis=0)

    bundle = jnp.sum(cmpx, axis=0)
    #convert the complex sum back to an angle
    bundle = cmpx_to_unitary(bundle)
    bundle = jnp.reshape(bundle, (1, -1))

    return bundle

#unbundling operation - number of symbols bundled to form x
# is assumed from 'symbols' matrix or passed manually by n
# note - this operation is highly approximate
def unbundle(x, symbols, n=-1):
    #assume that the number of symbols bundled to form x is the number
    #passed in the matrix plus one
    if n <= 0:
        n = symbols.shape[0] + 1
    assert n >= symbols.shape[0], "Too many symbols to unbundle given this weight"
    
    x_cmpx = unitary_to_cmpx(x) * n
    s_cmpx = unitary_to_cmpx(symbols)

    symbol = x_cmpx - jnp.sum(s_cmpx, axis=0)
    symbol = cmpx_to_unitary(symbol)
    symbol = jnp.reshape(symbol, (1, -1))

    return symbol

#bind a list of symbols together
def bind_list(*symbols):
    #stack the vararg inputs into an array
    symbols = jnp.stack(symbols)

    return bind(symbols)

#binding operation for FHRR
def bind(symbols):
    #sum the angles
    symbol = jnp.sum(symbols, axis=0)
    #remap the angles to (-1, 1)
    symbol = remap_phase(symbol)
    #reshape the output to maintain 2D array
    symbol = jnp.reshape(symbol, (1, -1))

    return symbol

#unbind a list of symbols from another symbol, x
def unbind_list(x, *symbols):
    #stack and sum the symbols to be unbound
    symbols = jnp.stack(symbols, axis=0)

    return unbind(x, symbols)

#unbinding operation for FHRR
def unbind(x, symbols):
    symbols = jnp.sum(symbols, axis=0)

    #remove them from the input & remap phase
    symbol = jnp.subtract(x, symbols)
    symbol = remap_phase(symbol)

    return symbol