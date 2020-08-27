with import ./nix/nixpkgs.nix {

  overlays = [

    (self: super:
      {
        blas = super.blas.override {
          blasProvider = self.mkl;
        };
        lapack = super.lapack.override {
          lapackProvider = self.mkl;
        };
      }
    )
  ];

};

let
  py = python3;
in
mkShell {
  buildInputs = [

    ffmpeg
    entr

    (py.withPackages (ps: with ps; [

      # unagan dependencies
      librosa
      # ffmpeg-full leads to assertion error with blas override.
      (pydub.override { ffmpeg-full = ffmpeg; })
      click
      pyyaml
      pytorchWithCuda
      scipy

      pip

      # dev deps
      pudb  # debugger
      black
      ipython
      pyls-isort
      pyls-black
      pyls-mypy
      python-language-server
    ]))
   ];

  shellHook = ''
    export PIP_PREFIX="$(pwd)/.build/pip_packages"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export PYTHONPATH="$PIP_PREFIX/${py.sitePackages}:$PYTHONPATH"
    unset SOURCE_DATE_EPOCH
  '';
}
