using Documenter, FunctionProperties

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

pages = [
    "Home" => "index.md",
    "design.md",
    "api.md",
]

makedocs(
    sitename = "FunctionProperties.jl",
    authors = "Chris Rackauckas",
    modules = [FunctionProperties],
    clean = true, doctest = true, linkcheck = true,
    format = Documenter.HTML(
        analytics = "UA-90474609-3",
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/FunctionProperties/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/FunctionProperties.jl.git";
    push_preview = true
)
