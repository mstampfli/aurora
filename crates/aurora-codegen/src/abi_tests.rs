//! The backend's half of the builtin-table contract: what the JIT/AOT module
//! actually declares must be exactly what `aurora-abi` says, so a table row and
//! the code generated from it can never describe different functions.

use super::*;

/// The host imports of a real (empty) compilation, keyed the way call lowering
/// looks them up, paired with the symbol and Cranelift signature each declares,
/// plus the target's pointer type (what a `Ptr` parameter lowers to).
fn declared_imports() -> (Type, Vec<(&'static str, String, Vec<Type>, Vec<Type>)>) {
    let (module, diags) = aurora_parser::parse_str("fn main() { }");
    assert!(!diags.iter().any(|d| d.is_error()), "parse failed");
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    register_host_symbols(&mut builder);
    let mut jmod = JITModule::new(builder);
    let m = monomorphized(&module).expect("monomorphize failed");
    let (env, _) = lower(&m, &mut jmod, false, false, false, Vec::new()).expect("lower failed");
    let mut out: Vec<_> = env
        .hosts
        .iter()
        .map(|(&key, &id)| {
            let decl = jmod.declarations().get_function_decl(id);
            (
                key,
                decl.linkage_name(id).into_owned(),
                decl.signature
                    .params
                    .iter()
                    .map(|p| p.value_type)
                    .collect::<Vec<_>>(),
                decl.signature
                    .returns
                    .iter()
                    .map(|p| p.value_type)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    out.sort_by_key(|(k, ..)| *k);
    (jmod.target_config().pointer_type(), out)
}

#[test]
fn host_imports_match_the_builtin_table() {
    let (ptr, declared) = declared_imports();
    let clif = |t: aurora_abi::Ty| match t {
        aurora_abi::Ty::I64 => types::I64,
        aurora_abi::Ty::F64 => types::F64,
        aurora_abi::Ty::Ptr => ptr,
        // Never reached: `abi_params`/`abi_ret` turn a `Str` result into the
        // leading out-pointer, so no declared slot is ever `Str`.
        aurora_abi::Ty::Str => unreachable!("Str is not an ABI slot type"),
    };

    let expected: Vec<&aurora_abi::Builtin> = aurora_abi::TABLE
        .iter()
        .filter(|b| b.kind.is_host_import())
        .collect();
    let declared_keys: Vec<&str> = declared.iter().map(|(k, ..)| *k).collect();
    let mut expected_keys: Vec<&str> = expected.iter().map(|b| b.name).collect();
    expected_keys.sort();
    assert_eq!(
        declared_keys, expected_keys,
        "the backend's host table is not the builtin table"
    );

    for (key, symbol, params, returns) in &declared {
        let row = aurora_abi::lookup(key).expect("declared import is not in the table");
        assert_eq!(symbol, row.symbol, "`{key}` imports the wrong symbol");
        let want: Vec<Type> = row.abi_params().iter().map(|t| clif(*t)).collect();
        assert_eq!(params, &want, "`{key}` imports the wrong parameter types");
        let want_ret: Vec<Type> = row.abi_ret().map(clif).into_iter().collect();
        assert_eq!(returns, &want_ret, "`{key}` imports the wrong return type");
    }
}

/// The call site coerces each argument to the type `scalar_builtin_sig` gives
/// and then calls the import declared above. If those two disagreed the
/// generated call would pass a float where the callee reads an integer.
#[test]
fn scalar_dispatch_agrees_with_the_declared_import() {
    let mut checked = 0;
    for (key, _, params, returns) in declared_imports().1 {
        let Some((sig_params, sig_ret)) = scalar_builtin_sig(key) else {
            continue;
        };
        let clif = |t: aurora_abi::Ty| match t {
            aurora_abi::Ty::F64 => types::F64,
            aurora_abi::Ty::I64 | aurora_abi::Ty::Ptr => types::I64,
            aurora_abi::Ty::Str => unreachable!("a scalar row cannot return Str"),
        };
        let want: Vec<Type> = sig_params.iter().map(|t| clif(*t)).collect();
        assert_eq!(
            params, want,
            "`{key}` is called with types its import does not take"
        );
        let want_ret: Vec<Type> = sig_ret.map(clif).into_iter().collect();
        assert_eq!(
            returns, want_ret,
            "`{key}` returns a type its import does not return"
        );
        // The compiled type the call site yields must follow the same mapping.
        assert_eq!(sig_ret.map(abi_cty), sig_ret.map(|t| clif_to_cty(clif(t))));
        checked += 1;
    }
    assert_eq!(
        checked,
        aurora_abi::TABLE
            .iter()
            .filter(|b| b.kind == aurora_abi::Kind::Scalar)
            .count()
    );
}

fn clif_to_cty(t: Type) -> Cty {
    if t == types::F64 {
        Cty::F64
    } else {
        Cty::I64
    }
}

/// Every builtin Aurora source can call is either lowered inline or backed by an
/// import: a name the front end accepts but the backend has no callee for would
/// stub the whole enclosing function.
#[test]
fn every_visible_builtin_is_reachable() {
    let declared: Vec<&str> = declared_imports().1.iter().map(|(k, ..)| *k).collect();
    for b in aurora_abi::TABLE
        .iter()
        .filter(|b| b.kind.is_aurora_visible())
    {
        if b.kind == aurora_abi::Kind::Inline {
            continue;
        }
        assert!(
            declared.contains(&b.name),
            "builtin `{}` has no host import",
            b.name
        );
    }
}
