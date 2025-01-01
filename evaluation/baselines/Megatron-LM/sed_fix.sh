sed -i "s/from pkg_resources import packaging/from pkg_resources.extern import packaging/g" `grep -rl "from pkg_resources import packaging"`
