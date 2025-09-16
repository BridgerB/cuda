{
  description = "CUDA Mandelbrot Set Rendering with PPM to PNG conversion and browser display";

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

      mandelbrot = pkgs.stdenv.mkDerivation {
        pname = "mandelbrot";
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
          nvcc -o mandelbrot mandelbrot.cu -lcudart
        '';

        installPhase = ''
          mkdir -p $out/bin
          cp mandelbrot $out/bin/
        '';
      };

      # Wrapper script to run mandelbrot, convert PPM to PNG, and open in Chromium
      runScript = pkgs.writeShellScriptBin "mandelbrot-run" ''
        #!/bin/sh
        ${mandelbrot}/bin/mandelbrot
        ${pkgs.imagemagick}/bin/convert mandelbrot.ppm mandelbrot.png
        ${pkgs.chromium}/bin/chromium mandelbrot.png
      '';

    in
    {
      packages.x86_64-linux.default = mandelbrot;
      apps.x86_64-linux.default = {
        type = "app";
        program = "${runScript}/bin/mandelbrot-run";
      };
    };
}