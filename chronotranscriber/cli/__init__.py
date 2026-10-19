"""Command-line front end of ChronoTranscriber.

``cli.parser`` builds the top-level parser and one parser per subcommand
from the option tables; ``cli.transcribe``, ``cli.repair`` and ``cli.batch``
run the subcommands and ``cli.command`` holds their shared plumbing.
"""
