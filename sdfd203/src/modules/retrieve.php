<?php
// modules/retrieve.php, Jake Deery, 2018

// retrieve data from the database
function retrieve($input_arr) {
	foreach ($input_arr as $input_value) {
		// define connection details
		$con = mysqli_connect("127.0.0.1", "username", "passwd", "dbname");
		// check connection
		if (mysqli_connect_errno() == true) {
			printf("<P STYLE='color:red;margin:0'>ERROR: Connect failed: " . mysqli_connect_error() . "</P>\n");
			exit();
		}

		if ($input_value == "*"){
			$sql = "SELECT * FROM `axessta_sdfd203`.`searchSchema`";
		} else {
			$sql = "SELECT * FROM `axessta_sdfd203`.`searchSchema` WHERE keywords = '" . mysqli_escape_string($con, $input_value) . "'";
		}

		$result = mysqli_query($con, $sql);
		while($row = mysqli_fetch_assoc($result)) {
			$id_arr[$row["id"]] = $row["id"];
			$key_arr[$row["id"]] = $row["keywords"];
			$dir_arr[$row["id"]] = $row["directory"];
		}

		mysqli_close($con); // Closing connection
	}

	if (isset($id_arr) && isset($key_arr) && isset($dir_arr)) {
		array_multisort($key_arr, $dir_arr, $id_arr);
		return array("id"=>$id_arr, "key"=>$key_arr, "dir"=>$dir_arr);
	}
}
?>
