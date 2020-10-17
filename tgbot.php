<?php

    $token = '<BOT ID>';
    $j = json_decode(file_get_contents("php://input"));

    function appendFileUnique($fp, $line)
    {
        $data = file_get_contents($fp);
        if(strpos($data, $line . "\n") === false)
            file_put_contents($fp, $line . "\n", FILE_APPEND | LOCK_EX);
    }

    if(isset($j->{'message'}->{'text'}) && isset($j->{'message'}->{'chat'}->{'id'}))
    {
        if(strpos($j->{'message'}->{'text'}, "/quote") !== FALSE)
        {
            $file = file("out.txt"); 
            $line = $file[rand(0, count($file) - 1)];
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode($line));
            http_response_code(200);
            exit;
        }
        
        if(strpos($j->{'message'}->{'text'}, "/info") !== FALSE)
        {
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            $lines = explode(PHP_EOL, file_get_contents('botmsg.txt'));
            $bm = file_get_contents("portstat.txt");
            $p = strstr($bm, "Digest size: ");
            $p = substr($p, 13);
            $p = explode("\n", $p, 2)[0];
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode($bm . "\nI am " . number_format(count($lines)) . " / " . $p . " away from re-computing my network.\n"));
            http_response_code(200);
            exit;
        }

        $msg = $j->{'message'}->{'text'};

        $pp = explode(' ', $msg);
        $pps = array_slice($pp, 0, 16);

        $str = "";
        foreach($pps as $p)
            if(strlen($p) <= 250 && $p != "" && $p != " ")
                $str .= str_replace("\n", "", $p) . " ";
        $str = rtrim($str, ' ');

        appendFileUnique("botmsg.txt", str_replace("\n", "", substr($str, 0, 4090))); //you could reduce this to 768 or 1024 safely

        foreach($pp as $p)
            if(strlen($p) <= 250 && $p != "" && $p != " ")
                appendFileUnique("botdict.txt", str_replace("\n", "", substr($p, 0, 250)));
    }

    http_response_code(200);

?>
