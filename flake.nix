{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    jupyter.url = "github:kirelagin/jupyter.nix";
  };

  outputs = { self, nixpkgs, flake-utils, jupyter }:
    flake-utils.lib.eachDefaultSystem (system: {
      packages = {
        jupyter = jupyter.lib.makeJupyterLab {
          pkgs = nixpkgs.legacyPackages.${system};
          kernels = {
            "python3".ipykernel = {
              packages = pp: with pp; [
                numpy
                polars
              ];
              withPlotly = true;
            };
          };
        };
      };
    });
}