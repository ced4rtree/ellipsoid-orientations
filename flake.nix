{
  description = "Ellipsoid Testing Code";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
  };

  outputs = { nixpkgs, ... }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        conda
      ];

      shellHook = ''
        conda-shell
        conda activate ellipsoids
      '';
    };
  };
}
