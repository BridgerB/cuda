{
  description = "CUDA Bitcoin Miner";

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

    bitcoinMiner = pkgs.stdenv.mkDerivation {
      pname = "bitcoin-cuda-miner";
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
        nvcc -o bitcoin_miner main.cu bitcoin_miner.cu -lcudart -O3 --ptxas-options=-v
      '';

      installPhase = ''
        mkdir -p $out/bin
        cp bitcoin_miner $out/bin/
      '';
    };

    # Test script to verify the miner works with known genesis block
    testScript = pkgs.writeShellScriptBin "test-miner" ''
      #!/bin/sh
      echo "=== Testing Bitcoin CUDA Miner with Genesis Block ==="
      echo
      echo "Expected result: nonce 2083236893"
      echo "Expected hash: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
      echo

      # Test with wider range first to see if we can find anything
      echo "Testing wide range around known solution..."
      BLOCK_HEADER="0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d00000000"
      TARGET="00000000ffff0000000000000000000000000000000000000000000000000000"

      ${bitcoinMiner}/bin/bitcoin_miner "$BLOCK_HEADER" 2083200000 2084000000 "$TARGET"

      echo
      echo "If not found, let's verify our block header parsing..."
      echo "Block header length: ''${#BLOCK_HEADER} characters (should be 160)"
      echo "Target length: ''${#TARGET} characters (should be 64)"
    '';

    # Hash verification tool
    verifyTool = pkgs.stdenv.mkDerivation {
      pname = "verify-hash";
      version = "1.0.0";
      src = ./.;

      buildPhase = ''
        gcc -o verify_hash verify_hash.cu -std=c99
      '';

      installPhase = ''
        mkdir -p $out/bin
        cp verify_hash $out/bin/
      '';
    };

    # Debug script to test parsing
    debugScript = pkgs.writeShellScriptBin "debug-miner" ''
      #!/bin/sh
      echo "=== Debug Bitcoin CUDA Miner ==="
      echo

      # First let's verify what we expect to see
      echo "Step 1: Expected Genesis Block Information"
      echo "Expected nonce: 2083236893 (0x7c2bac1d)"
      echo "Expected hash:  000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
      echo "Expected serialized header:"
      echo "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c"
      echo

      # Verify with system tools
      echo "Step 2: System SHA-256 verification"
      # Create the exact header bytes for verification
      printf '\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3b\xa3\xed\xfd\x7a\x7b\x12\xb2\x7a\xc7\x2c\x3e\x67\x76\x8f\x61\x7f\xc8\x1b\xc3\x88\x8a\x51\x32\x3a\x9f\xb8\xaa\x4b\x1e\x5e\x4a\x29\xab\x5f\x49\xff\xff\x00\x1d\x1d\xac\x2b\x7c' > /tmp/genesis_header.bin
      echo -n "System double SHA-256: "
      ${pkgs.coreutils}/bin/sha256sum /tmp/genesis_header.bin | ${pkgs.coreutils}/bin/cut -d' ' -f1 | ${pkgs.coreutils}/bin/xxd -r -p | ${pkgs.coreutils}/bin/sha256sum | ${pkgs.coreutils}/bin/cut -d' ' -f1
      rm -f /tmp/genesis_header.bin
      echo

      # Test with a very small range to see exact output
      BLOCK_HEADER="0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d00000000"
      TARGET="00000000ffff0000000000000000000000000000000000000000000000000000"

      echo "Step 3: Test CUDA miner with exact nonce 2083236893..."
      ${bitcoinMiner}/bin/bitcoin_miner "$BLOCK_HEADER" 2083236893 2083236893 "$TARGET"

      echo
      echo "Step 4: Test small range around it..."
      ${bitcoinMiner}/bin/bitcoin_miner "$BLOCK_HEADER" 2083236890 2083236900 "$TARGET"
    '';
  in {
    packages.x86_64-linux.default = bitcoinMiner;
    packages.x86_64-linux.test = testScript;
    packages.x86_64-linux.debug = debugScript;
    apps.x86_64-linux.default = {
      type = "app";
      program = "${bitcoinMiner}/bin/bitcoin_miner";
    };
    apps.x86_64-linux.test = {
      type = "app";
      program = "${testScript}/bin/test-miner";
    };
    apps.x86_64-linux.debug = {
      type = "app";
      program = "${debugScript}/bin/debug-miner";
    };
  };
}
