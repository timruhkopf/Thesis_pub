#!/bin/bash

# open up browser (and continue in non blocking state :&)
#detach xdg-open http://127.0.0.1:16006/ #&

python -m webbrowser -t "http://127.0.0.1:16006/"

# kill port 6007 (preventiv, to make it avail to TB)
ssh -L 16006:127.0.0.1:6006 truhkop@gwdu103.gwdg.de <~/PycharmProjects/Thesis/shell/remote_tb.sh
