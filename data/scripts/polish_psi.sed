#!/bin/sed -f

/\.frame/d
s/\.ite/ite/g
/(check-sat)/d
/(exit)/d
