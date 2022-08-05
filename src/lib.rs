mod bindings;

pub use bindings::*;

macro_rules! vdf_impl {
    (
        $VDF:ident,
        $minroot:ident,
        $minroot_verify:ident,
    ) => {
        pub struct $VDF {}

        impl $VDF {
            pub fn minroot() {}
            pub fn minroot_verify() {}
        }
    };
}

vdf_impl!(PallasVDF, minroot_pallas, minroot_verify_pallas,);
vdf_impl!(VestaVDF, minroot_vesta, minroot_verify_vesta,);
