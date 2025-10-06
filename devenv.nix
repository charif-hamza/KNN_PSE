{ pkgs, lib, config, inputs, ... }:

{
  env.GREET = "devenv";

  packages = [
    pkgs.git
    pkgs.zlib                 # libz.so.1
    pkgs.stdenv.cc.cc.lib     # libstdc++.so.6, libgcc_s.so.1
    pkgs.gfortran.cc.lib      # libgfortran for numpy/scipy
  ];

languages.python = {
    enable = true;
    package = pkgs.python312;           # <- avoid 3.13
    venv.enable = true;
    venv.requirements = ''
      --index-url https://pypi.org/simple
      --extra-index-url https://download.pytorch.org/whl/cpu
      torch
      numpy
      scipy
      pandas
      scikit-learn
      matplotlib
      pyarrow
      python-pptx 
      beautifulsoup4 
      lxml 
      seaborn
      tqdm
    '';
  };


  enterShell = ''
    echo hello from $GREET
    git --version
    python -V
  '';
}

