{
  description = "CUDA SHA-256 Hash Computation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };

    sha256Cuda = pkgs.stdenv.mkDerivation {
      pname = "sha256-cuda";
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
        nvcc -o sha256 sha256.cu -lcudart
      '';

      installPhase = ''
        mkdir -p $out/bin
        cp sha256 $out/bin/
      '';
    };

    # Wrapper script to run SHA-256 and compare with system command
    runScript = pkgs.writeShellScriptBin "sha256-run" ''
      #!/bin/sh
      echo "=== CUDA SHA-256 vs System sha256sum Comparison ==="
      echo

      # Run CUDA program
      ${sha256Cuda}/bin/sha256

      echo
      echo "=== System Verification ==="

      # Create test files for system sha256sum
      echo -n "Hello CUDA SHA-256!" > /tmp/test_base.txt

      # Test the base string (should match nothing since GPU adds index)
      echo "Base string SHA-256:"
      echo -n "Hello CUDA SHA-256!" | ${pkgs.coreutils}/bin/sha256sum

      # Create the string with index 0 appended (4 bytes: 0x00000000)
      printf "Hello CUDA SHA-256!\x00\x00\x00\x00" > /tmp/test_with_index0.bin
      echo
      echo "String + index 0 (matches GPU hash 0):"
      ${pkgs.coreutils}/bin/sha256sum /tmp/test_with_index0.bin

      # Create the string with index 1 appended (4 bytes: 0x01000000 little endian)
      printf "Hello CUDA SHA-256!\x01\x00\x00\x00" > /tmp/test_with_index1.bin
      echo "String + index 1 (matches GPU hash 1):"
      ${pkgs.coreutils}/bin/sha256sum /tmp/test_with_index1.bin

      # Clean up
      rm -f /tmp/test_base.txt /tmp/test_with_index0.bin /tmp/test_with_index1.bin

      echo
      echo "=== Verification Complete ==="
      echo "Compare the GPU hash 0 and hash 1 with the system results above!"
    '';
  in {
    packages.x86_64-linux.default = sha256Cuda;
    apps.x86_64-linux.default = {
      type = "app";
      program = "${runScript}/bin/sha256-run";
    };
  };
}
