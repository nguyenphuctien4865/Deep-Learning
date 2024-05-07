import java.util.Arrays;
import java.util.List;
import java.util.stream.Collector;
import java.util.stream.Collectors;

public class Main {
	public static void main(String[] args) {
		String[] words = new String[] {"hello", "world", "java", "test"};
		List<String> list = Arrays.stream(words)
			.map(w -> w.concat("1")).collect(Collectors.toList());
		System.out.println(list);
	}
}

