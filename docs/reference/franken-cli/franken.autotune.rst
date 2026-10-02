Autotune
========

.. argparse::
    :module: franken.autotune.script
    :func: get_parser_fn
    :prog: franken.autotune
    :nodefault:

Command builder
---------------

For a guided UI with defaults, option explanations and a copyable shell command,
open the `autotune command builder <../../_static/autotune.html>`_.
The standalone page validates required fields and supported value formats locally.
It does not check whether files, registered datasets or checkpoints exist.

From a source checkout with Franken installed, regenerate the page after CLI
changes with::

    python scripts/autotune_ui.py

For live validation through the actual Python CLI parser, run::

    python scripts/autotune_ui.py --serve

Then open ``http://127.0.0.1:8765``. The server only validates configuration;
it does not load models or datasets or launch training. Stop it with Ctrl+C.
Use ``--output path/to/autotune.html`` or ``--port 8766`` to change the output
location or server port. The UI uses no external JavaScript libraries or Node.

To share the builder with colleagues on your local network, bind to all IPv4
interfaces::

    python scripts/autotune_ui.py --serve --host 0.0.0.0

Colleagues can then open ``http://<this-machine-ip>:8765``. Use the machine's
LAN IP address in the URL; ``0.0.0.0`` is the bind address. The default host
remains ``127.0.0.1``. If the connection is blocked, allow inbound TCP port
8765 in your firewall.
