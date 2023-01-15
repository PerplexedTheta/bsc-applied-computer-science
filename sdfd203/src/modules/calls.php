<?php
// modules/calls.php, Jake Deery, 2018
if(!function_exists("check_db")) {
	include "check_db.php";
}

if(!function_exists("truncate")) {
	include "truncate.php";
}

if(!function_exists("update")) {
	include "update.php";
}

// run through and execute the specified function
if (isset($_GET["check_db"])) {
	check_db();
} elseif (isset($_GET["truncate"])) {
	truncate();
} elseif (isset($_GET["update"])) {
	update();
}

?>