class Perceptron:

    def __init__(self, *weights):
        # input: learning rate n, weight arguments corresponding to w_0 ... w_n
 
        # makes sure weights are passed as args, not as a list
        for w in weights:
            try: iter(w)
            except TypeError: pass
            else: raise TypeError(f"recieved an iterable '{w}' as a weight, expected all weights to be numerical")

        self.weights = tuple([w for w in weights])

    def o(self, *inputs):
        # input: inputs as args e.g. my_perceptron.o(x1, x2,...,xn)

        # makes sure inputs are passed as args, not as a list
        for x in inputs:
            try: iter(x)
            except TypeError: pass
            else: raise TypeError(f"recieved an iterable '{x}' as a input, expected all inputs to be numerical")

        n_expected_inputs = len(self.weights)-1 # number of inputs expected
        n_recieved_inputs = len(inputs) # number of inputs recieved

        if n_expected_inputs != n_recieved_inputs: 
            # make sure inputs is correct length
            expected_str = f"x_1...x_{n_expected_inputs}"
            recieved_str = f'x_1...x_{n_recieved_inputs}' if n_recieved_inputs>1 else 'x_1' if n_recieved_inputs==1 else ''
            raise ValueError(f"expected ({expected_str}), got ({recieved_str})")

        inputs = (1,) + tuple(inputs) # tuple of inputs, first element is constant 1

        def sgn(x):
            # sign function
            return 1 if x > 0 else -1

        def dotprod(avec, bvec): 
            # dot product of two 1D vectors
            return sum([a*b for a,b in zip(avec, bvec)])

        return sgn( dotprod(self.weights, inputs) ) # perceptron output o