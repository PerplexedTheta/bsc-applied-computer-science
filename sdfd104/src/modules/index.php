<?php
header('HTTP/1.0 403 Forbidden');
readfile("/var/http-aliases/errors/403.html");
?>
