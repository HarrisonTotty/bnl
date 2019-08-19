//! A machine learning thingy.

pub mod network;

fn main() {
    let n = network::Network::new(6, vec![6, 7, 6]);
    let input = vec![true, false, true, true, false, true];

    println!("n = {:?}\n", n);
    println!("Result = {:?}", n.apply(input));
}
