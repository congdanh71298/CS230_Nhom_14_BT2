def log_to_file(logfile, message):
    with open(logfile, "a") as f:
        f.write(message + "\n")
