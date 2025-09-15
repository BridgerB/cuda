{
  description = "CUDA Vector Addition";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      vectorAdd = pkgs.stdenv.mkDerivation {
        pname = "vector-add";
        version = "1.0.0";
        src = ./.;

        nativeBuildInputs = with pkgs; [
          cudaPackages.cuda_nvcc
          addDriverRunpath
        ];

        buildInputs = with pkgs; [
          cudaPackages.cuda_cudart
          linuxPackages.nvidia_x11
        ];

        buildPhase = ''
          nvcc -o vector_add vector_add.cu -lcudart
        '';

        installPhase = ''
          mkdir -p $out/bin
          cp vector_add $out/bin/
        '';
      };

    in
    {
      packages.x86_64-linux.default = vectorAdd;
      apps.x86_64-linux.default = {
        type = "app";
        program = "${vectorAdd}/bin/vector_add";
      };
    };
}