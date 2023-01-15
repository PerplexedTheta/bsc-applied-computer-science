<!-- Modules: signup.php - Jake Deery -->
<?php
// signup.php - Jake Deery 2017
if($_POST){
    // define connection details
	$con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");

	// check connection
	if (mysqli_connect_errno()) {
		printf("Connect failed: %s\n", mysqli_connect_error());
		exit();
	}

	$accountName = $_POST['accountName'];
	$password = $_POST['password'];

	$accountName = mysqli_real_escape_string($con, $accountName);
	$password = mysqli_real_escape_string($con, $password);

	$query = mysqli_query($con, "INSERT INTO `account` (`accountName`, `password`, `role`, `online`, `banned`) VALUES ('$accountName', '$password', 000001, 0, 0)");

	if ($query == TRUE) {
		echo "<h3>Submission success! Please <a href='./'>return home</a>.</h3>";
	}

	mysqli_close($con);
} ?>
