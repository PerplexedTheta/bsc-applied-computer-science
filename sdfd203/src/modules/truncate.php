<?php
// modules/truncate.php, Jake Deery, 2018

// empty the database of contents
function truncate() {
	// define connection details
	$con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");
	// check connection
	if (mysqli_connect_errno() == true) {
		printf("<SPAN STYLE='color:red;margin:0'>ERROR: Connect failed: " . mysqli_connect_error() . "</SPAN>\n");
		exit();
	}

	// emptry the table of existing rows
	$truncate = "TRUNCATE TABLE `axessta_sdfd203`.`searchSchema`;";
	if (!mysqli_query($con, $truncate)) {
		printf("<SPAN STYLE='color:red;margin:0'>ERROR: " . $truncate . "<BR />"  . mysqli_error($con) . "</SPAN>\n");
		exit();
	} else {
		printf("<SPAN STYLE='color:green;margin:0'>INFO: The database was truncated without error.</SPAN>\n");
	}

	mysqli_close($con); // Closing connection
}
?>