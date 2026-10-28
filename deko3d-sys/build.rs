fn main() {
    println!("cargo:rerun-if-env-changed=DEVKITPRO");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("horizon") {
        return;
    }
    let dkp = std::env::var("DEVKITPRO").expect("DEVKITPRO must point at the devkitPro install");
    println!("cargo:rustc-link-search=native={dkp}/libnx/lib");
    let lib = if std::env::var_os("CARGO_FEATURE_DEBUG").is_some() {
        "deko3dd"
    } else {
        "deko3d"
    };
    println!("cargo:rustc-link-lib=static={lib}");
}
