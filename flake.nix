{
  description = "CUDA Hello World program";

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

      cudaHello = pkgs.stdenv.mkDerivation {
          pname = "cuda-hello";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = with pkgs; [
            cudaPackages.cuda_nvcc
            addDriverRunpath
            gcc
          ];

          buildInputs = with pkgs; [
            cudaPackages.cuda_cudart
            linuxPackages.nvidia_x11
          ];

          buildPhase = ''
            nvcc -o hello hello.cu -lcudart
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp hello $out/bin/
          '';

        };

    in
    {
      packages.x86_64-linux.default = cudaHello;
      apps.x86_64-linux.default = {
        type = "app";
        program = "${cudaHello}/bin/hello";
      };
    };
}