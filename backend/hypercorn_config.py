

# Production server configuration

bind = "0.0.0.0:${PORT}"
workers = 2
keep_alive_timeout = 300
timeout = 120
accesslog = "-"  # Log to stdout