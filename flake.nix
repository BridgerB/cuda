{
  description = "CUDA Hello World program";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
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
            export CUDA_PATH=${pkgs.cudaPackages.cuda_nvcc}
            export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH

            # Use the real CUDA lib dir
            CUDA_LIB_DIR=${pkgs.cudaPackages.cuda_cudart}/lib

            nvcc -o hello hello.cu \
              -L"$CUDA_LIB_DIR" \
              -lcudart \
              -Wno-deprecated-gpu-targets
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp hello $out/bin/
          '';

          meta = with pkgs.lib; {
            description = "CUDA Hello World program";
            platforms = platforms.linux;
          };
        };

      in
      {
        packages = {
          default = cudaHello;
          cuda-hello = cudaHello;
        };

        apps = {
          default = {
            type = "app";
            program = "${cudaHello}/bin/hello";
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cudatoolkit
            linuxPackages.nvidia_x11
            gcc
          ];

          shellHook = ''
            export CUDA_PATH=${pkgs.cudaPackages.cuda_nvcc}
            export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
          '';
        };
      });
}