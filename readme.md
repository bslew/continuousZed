# GENERAL

Package to process slowly varying zenith distance pointing corrections of the 32-m radio telescope
in Toruń.

This program alleviates the problems modeling aging effects of the RT-32 associated with 
gradually increasing sag of the secondary mirror leading to non-zero 
zenith distance pointing offsets.


# DOWNLOAD

```
git clone https://github.com/bslew/continuousZed.git
```

# INSTALL

Download the package from git repository

## Installation steps

Change directory to newly downloaded continuousZed repository, create and activate virtual environment,
update it and install required packages.

```
cd continuousZed
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

Execute

```
make install
```

You can edit ~/.config/continuousZed/continuousZed.ini config file to match your preferences.

# Use

Use continuousZed.py from virtual environment, i.e activate the environment unless you already have done so:

```
cd continuousZed
. venv/bin/activate
```

"(venv)" prompt will appear to indicate you are in the virtual environment.

```
continuousZed.py --help
```

# Examples

To calculate and set ZD correction using the last 90 days median type:

```
continuousZed.py --setauto
```

To set the correction to any other value type eg.:

```
continuousZed.py --set -0.020
```

To null corrections type:

```
continuousZed.py --set 0
```


# AUTHOR
Bartosz Lew [<bartosz.lew@umk.pl>](bartosz.lew@umk.pl)

