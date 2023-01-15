<?php
// modules/check_db.php, Jake Deery, 2018

// check the database is working
function check_db() {
	// define connection details
	$con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");
	// check connection
	if (mysqli_connect_errno() == true) {
		printf("<SPAN STYLE='color:red;margin:0'>ERROR: Connect failed: " . mysqli_connect_error() . "</SPAN>\n");
		exit();
	} else {
		printf("<SPAN STYLE='color:green;margin:0'>INFO: The database is online and accessable.</SPAN>\n");
	}

	mysqli_close($con); // Closing connection
}
?>