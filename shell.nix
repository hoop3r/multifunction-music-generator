  
  # Ran into some funky package compatibility issues...
  # ...so I had to build a couple from source myself so they'd play nice.
  
  # All good and working now! :D 

with import <nixpkgs> {};

let
  pythonEnv = python3Packages.python.withPackages (ps: with ps; [
    ipykernel
    jupyterlab
    numpy
    pandas
    wxpython
    setuptools
    wheel
    pyoPkg
    sounddevice
    midiutil
    matplotlib
  ]);

  liblo28 = stdenv.mkDerivation rec {
    pname = "liblo";
    version = "0.28";

    src = fetchurl {
      url = "https://download.sourceforge.net/liblo/liblo-0.28.tar.gz";
      sha256 = "2pSptnuTYlNU3Yn/f+MeUpf8lAC26vc3jILuHK99uQk=";
    };

    nativeBuildInputs = [ pkg-config autoconf automake libtool ];

    configurePhase = ''
      ./configure --prefix=$out
    '';

    buildPhase = ''
      make
    '';

    installPhase = ''
      make install
    '';
  };

  pyoPkg = python3Packages.buildPythonPackage rec {
    pname = "pyo";
    version = "1.0.5";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/77/c8/e949d16170a9f448994be74963fad54557c13d1c4e4302590fa35280ae55/pyo-1.0.5.tar.gz";
      sha256 = "4ELZR6C2QbQA4ij54h7soh34v0iVxtvQE/h2ONdyjjE=";
    };

    nativeBuildInputs = [ pkg-config gcc ];

    propagatedBuildInputs = [ liblo28 portaudio libsndfile fftw portmidi ];

    doCheck = false;
  };

in
mkShell {
  name = "mmg-dev";

  buildInputs = [
    pythonEnv

    portaudio
    portmidi
    libsndfile
    alsa-lib
    alsa-utils
    fftw
    liblo28
    mesa
    libGL
    pkg-config
    gcc
    tk
    gtk3
    glib
    cairo
    libpng
    libjpeg_turbo
    lcms2
  ];

  shellHook = ''
    PYTHON_BIN="${pythonEnv}/bin/python3"

    echo "Now entering project dev shell using: ${pythonEnv}"

    echo "Jupyter kernel setup time yippee...!"
    "$PYTHON_BIN" -m ipykernel install --user \
      --name mmg-nix \
      --display-name "Python (mmg-nix)"

    echo "ready 2 go :P"
  '';
}