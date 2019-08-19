//! Contains the definition of components within a `bnl` network.

/// Represents a single layer of neurons in a `bnl` network.
#[derive(Clone,Debug)]
pub struct Layer {
    /// The collection of neurons present in this layer.
    pub neurons: Vec<Neuron>
}

/// Implements custom functions on `bnl` layers.
impl Layer {
    /// "Applies" this layer to a given input vector of boolean values.
    pub fn apply(&self, input: Vec<bool>) -> Vec<bool> {
        self.neurons.iter().map(|n| n.apply(input.clone())).collect()
    }

    /// Creates a new randomized layer of the specified input length and number
    /// of neurons.
    pub fn new(input_len: usize, num_neurons: usize) -> Self {
        let mut n: Vec<Neuron> = Vec::new();
        for _i in 0..num_neurons {
            n.push(Neuron::new(input_len));
        };
        Layer {
            neurons: n
        }
    }
}

/// Represents a `bnl` network.
#[derive(Clone,Debug)]
pub struct Network {
    /// The collection of layers present in this network.
    pub layers: Vec<Layer>
}

/// Implement custom methods for `bnl` networks.
impl Network {
    /// "Applies" this network on the specified input vector of boolean values.
    pub fn apply(&self, input: Vec<bool>) -> Vec<bool> {
        let mut res: Vec<bool> = input.clone();
        for layer in &self.layers {
            res = layer.apply(res);
        }
        res
    }
    
    /// Creates a new randomized network of the specified input length and
    /// vector of layer lengths (number of neurons in each layer).
    pub fn new(input_len: usize, layer_lengths: Vec<usize>) -> Self {
        let mut l: Vec<Layer> = Vec::new();
        for i in 0..layer_lengths.len() {
            if i == 0 {
                l.push(Layer::new(input_len, layer_lengths[i]));
            } else {
                l.push(Layer::new(layer_lengths[i - 1], layer_lengths[i]));
            }
        }
        Network {
            layers: l
        }
    }
}

/// Represents a single neuron within a `bnl` network.
#[derive(Clone,Debug)]
pub struct Neuron {
    /// The bias of this neuron as a boolean value.
    pub bias: bool,
    
    /// A vector containing the boolean function input combinator numbers.
    /// The way that this works is by applying the specified boolean function
    /// to the result of each stage.
    pub input_combinators: Vec<u8>,

    /// The "result" combinator of this neuron (the function to apply between
    /// the initial result and the bias).
    pub result_combinator: u8
}

/// Implements custom methods available to `Neuron` structures.
impl Neuron {
    /// "Applies" this neuron to a given input vector of boolen values.
    pub fn apply(&self, input: Vec<bool>) -> bool {
        self.apply_result(self.apply_input(input))
    }

    /// "Applies" only the input combinator of this neuron to a given input
    /// vector of boolean values.
    pub fn apply_input(&self, input: Vec<bool>) -> bool {
        zip_combinator(input[0], input[1..].to_vec(), &self.input_combinators)
    }

    /// "Applies" the result combinator of this neuron to a given input boolean
    /// value and the neuron's own bias.
    pub fn apply_result(&self, input: bool) -> bool {
        compute_boolean(input, self.bias, self.result_combinator)
    }

    /// Creates a new randomized neuron with the given input vector length.
    pub fn new(input_len: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut ic: Vec<u8> = Vec::new();
        for _i in 0..input_len {
            ic.push(rng.gen_range(0, 16));
        }
        Neuron {
            bias: rng.gen(),
            input_combinators: ic,
            result_combinator: rng.gen_range(0, 16)
        }
    }
}

/// Computes the result of the specified boolean combinator on two input boolean
/// values.
pub fn compute_boolean(left: bool, right: bool, combinator: u8) -> bool {
    match combinator {
        0  => false,
        1  => left && right,
        2  => left && !right,
        3  => left,
        4  => !left && right,
        5  => right,
        6  => (left && !right) || (!left && right),
        7  => left || right,
        8  => !(left || right),
        9  => (left && right) || (!left && !right),
        10 => !right,
        11 => left || !right,
        12 => !left,
        13 => !left || right,
        14 => !(left && right),
        _  => true
    }
}

/// "Zips" and input vector across an input combinator vector.
pub fn zip_combinator(left: bool, remaining: Vec<bool>, combinators: &Vec<u8>) -> bool {
    match remaining.len() {
        0 => panic!("This shouldn't happen!"),
        1 => compute_boolean(left, remaining[0], combinators[0]),
        _ => compute_boolean(left, zip_combinator(remaining[0], remaining[1..].to_vec(), &combinators[1..].to_vec()), combinators[0])
    }
}
